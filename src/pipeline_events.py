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

import json
import logging
import os
import random
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .logging_templates import get_correlation_id, _format_correlation

logger = logging.getLogger(__name__)

# Event hook callback type: receives a PipelineEvent
EventCallback = Callable[['PipelineEvent'], None]


@dataclass
class PipelineEvent:
    """Structured event emitted during pipeline lifecycle.

    Attributes:
        event_type: One of 'before_stage', 'after_stage', 'on_stage_error',
                    'on_pipeline_complete', 'stage_start', 'stage_skip',
                    'checkpoint_save', 'resource_warning'
        stage_name: Name of the stage (empty string for pipeline-level events)
        timestamp: Unix timestamp when the event was created
        data: Arbitrary data dict (e.g., elapsed time, metrics, estimated_duration)
        error: Optional error string if the event relates to a failure
    """
    event_type: str
    stage_name: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_trace_dict(self) -> Dict[str, Any]:
        """Convert event to trace dictionary format for JSON logging.

        Returns:
            Dictionary with timestamp_ms, stage_name, duration_ms, and metadata_dict.
        """
        # US-159-005: Include correlation_id in trace output
        correlation_id = self.data.get('correlation_id', '-')
        return {
            'timestamp_ms': int(self.timestamp * 1000),
            'stage_name': self.stage_name,
            'event_type': self.event_type,
            'duration_ms': self.data.get('duration_ms') or self.data.get('elapsed', 0) * 1000,
            'correlation_id': correlation_id,
            'metadata_dict': self.data,
            'error': self.error,
        }


# Extended event types for enhanced pipeline monitoring
# These complement the core lifecycle events with additional monitoring capabilities
EVENT_STAGE_START = 'stage_start'  # Stage starting with estimated duration
EVENT_STAGE_PROGRESS = 'stage_progress'  # Stage progress update with percentage
EVENT_STAGE_SKIP = 'stage_skip'  # Stage skipped due to checkpoint
EVENT_CHECKPOINT_SAVE = 'checkpoint_save'  # Checkpoint saved
EVENT_RESOURCE_WARNING = 'resource_warning'  # CPU/memory threshold warning

# New event types for US-108-007 (Event Tracing)
EVENT_STAGE_CHECKPOINT_WRITE = 'stage_checkpoint_write'  # Checkpoint written during stage
EVENT_RESOURCE_THRESHOLD = 'resource_threshold'  # Resource usage threshold event
EVENT_ERROR_RATE_THRESHOLD = 'error_rate_threshold'  # US-138-010: Error rate threshold event
EVENT_ERROR_RECOVERED = 'error_recovered'  # Error was successfully recovered

# New event type for US-110-002 (Transcription Progress)
EVENT_TRANSCRIPTION_PROGRESS = 'transcription_progress'  # Transcription stage progress update

# New event types for US-120-004 (Detailed Pipeline Stage Tracing)
EVENT_STAGE_TRACE_START = 'stage_trace_start'  # Stage trace with full context (checkpoint state, matched count, memory)
EVENT_STAGE_TRACE_END = 'stage_trace_end'  # Stage trace end with duration and final context


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
                        'on_pipeline_complete', 'stage_start', 'stage_skip',
                        'checkpoint_save', 'resource_warning'
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
        # US-165-007: Log event emission for observability
        correlation_id = get_correlation_id()
        corr = _format_correlation(correlation_id)

        # Determine log level based on event type
        if event.event_type in ('on_stage_error', 'on_pipeline_error'):
            log_level = logging.ERROR
        elif event.event_type in ('on_pipeline_complete',):
            log_level = logging.INFO
        else:
            log_level = logging.DEBUG

        # Build context string from event data
        context_parts = []
        if event.data:
            for key in ['items_completed', 'items_total', 'progress_percent', 'elapsed', 'duration_ms', 'success']:
                if key in event.data:
                    context_parts.append(f"{key}={event.data[key]}")
        context_str = f" - {', '.join(context_parts)}" if context_parts else ""

        # Log the event emission
        logger.log(
            log_level,
            f"[EVENT:{event.event_type}] Stage: {event.stage_name}{corr}{context_str}"
        )

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


class EventTracer:
    """Event tracer that writes pipeline events to a JSON-lines file.

    Usage:
        tracer = EventTracer('/path/to/project/pipeline_trace.jsonl')
        tracer.start()
        bus.subscribe('before_stage', tracer.on_event)
        bus.subscribe('after_stage', tracer.on_event)
        # ... run pipeline ...
        tracer.stop()
    """

    def __init__(self, trace_file_path: str):
        """Initialize the tracer.

        Args:
            trace_file_path: Path to the JSON-lines trace file.
        """
        self.trace_file_path = trace_file_path
        self._file = None
        self._enabled = False

    def start(self) -> None:
        """Start tracing - opens the trace file for writing."""
        if self._enabled:
            return
        os.makedirs(os.path.dirname(self.trace_file_path), exist_ok=True)
        self._file = open(self.trace_file_path, 'w', encoding='utf-8')
        self._enabled = True
        logger.info(f"Event tracing started: {self.trace_file_path}")

    def stop(self) -> None:
        """Stop tracing - closes the trace file."""
        if not self._enabled:
            return
        if self._file:
            self._file.close()
            self._file = None
        self._enabled = False
        logger.info(f"Event tracing stopped: {self.trace_file_path}")

    def on_event(self, event: PipelineEvent) -> None:
        """Handler to log events to the trace file.

        Args:
            event: The PipelineEvent to trace.
        """
        if not self._enabled or not self._file:
            return
        try:
            trace_dict = event.to_trace_dict()
            self._file.write(json.dumps(trace_dict) + '\n')
            self._file.flush()
        except Exception as e:
            logger.warning(f"Failed to write trace event: {e}")


class EventReplay:
    """Utility to reconstruct pipeline execution timeline from trace file.

    Usage:
        replay = EventReplay('/path/to/project/pipeline_trace.jsonl')
        timeline = replay.load_timeline()
        analysis = replay.analyze()
    """

    def __init__(self, trace_file_path: str):
        """Initialize the replay utility.

        Args:
            trace_file_path: Path to the JSON-lines trace file.
        """
        self.trace_file_path = trace_file_path

    def load_timeline(self) -> List[Dict[str, Any]]:
        """Load all events from the trace file.

        Returns:
            List of event dictionaries in chronological order.
        """
        if not os.path.exists(self.trace_file_path):
            return []

        events = []
        with open(self.trace_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse trace line: {line}")
        return sorted(events, key=lambda e: e.get('timestamp_ms', 0))

    def analyze(self) -> Dict[str, Any]:
        """Analyze the trace to identify patterns.

        Returns:
            Dictionary with longest_stages, most_common_errors, retry_patterns.
        """
        events = self.load_timeline()
        if not events:
            return {
                'longest_stages': [],
                'most_common_errors': [],
                'retry_patterns': [],
            }

        # Find longest stages
        stage_durations: Dict[str, List[int]] = defaultdict(list)
        current_stage: str = ''
        stage_start_ms: int = 0
        for event in events:
            if event.get('event_type') == 'before_stage':
                current_stage = event.get('stage_name', '')
                stage_start_ms = event.get('timestamp_ms', 0)
            elif event.get('event_type') == 'after_stage' and current_stage:
                end_ms = event.get('timestamp_ms', 0)
                duration = end_ms - stage_start_ms
                stage_durations[current_stage].append(duration)
                current_stage = ''

        longest_stages = [
            {'stage': stage, 'total_duration_ms': sum(durations), 'count': len(durations)}
            for stage, durations in stage_durations.items()
        ]
        longest_stages.sort(key=lambda x: x['total_duration_ms'], reverse=True)

        # Find most common errors
        error_counts: Dict[str, int] = defaultdict(int)
        for event in events:
            if event.get('error'):
                error_counts[event['error']] += 1

        most_common_errors = [
            {'error': error, 'count': count}
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        # Find retry patterns (error_recovered events)
        retry_count = sum(1 for e in events if e.get('event_type') == EVENT_ERROR_RECOVERED)
        retry_patterns = [{'error_recovered_count': retry_count}]

        return {
            'longest_stages': longest_stages,
            'most_common_errors': most_common_errors,
            'retry_patterns': retry_patterns,
        }


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics.

    Returns:
        Dictionary with memory usage in MB: rss_mb, vms_mb, percent_used
    """
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024),
            'percent_used': process.memory_percent(),
        }
    except ImportError:
        # psutil not available
        return {'rss_mb': 0.0, 'vms_mb': 0.0, 'percent_used': 0.0}


def extract_checkpoint_summary(checkpoint_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a summary of checkpoint state for tracing.

    Args:
        checkpoint_data: Full checkpoint dictionary

    Returns:
        Summary dict with key counts and stage completion status
    """
    if not checkpoint_data:
        return {'has_checkpoint': False}

    summary = {
        'has_checkpoint': True,
        'version': checkpoint_data.get('version'),
        'last_completed_stage': checkpoint_data.get('last_completed_stage'),
    }

    # Count entries in each stage
    for stage_key in ['analyze', 'video_search', 'caption', 'match', 'iterative_match', 'download_segments']:
        stage_data = checkpoint_data.get(stage_key, {})
        if isinstance(stage_data, dict):
            summary[f'{stage_key}_has_data'] = len(stage_data) > 0

    # Match count
    match_data = checkpoint_data.get('match', {})
    if isinstance(match_data, dict):
        matches = match_data.get('matches', [])
        summary['match_count'] = len(matches) if isinstance(matches, list) else 0

    return summary


class TraceAnalyzer:
    """Enhanced trace analysis utilities for US-120-004.

    Provides detailed analysis of stage durations, resource usage,
    and checkpoint state changes during pipeline execution.

    Usage:
        analyzer = TraceAnalyzer('/path/to/pipeline_trace.jsonl')
        report = analyzer.generate_report()
    """

    def __init__(self, trace_file_path: str):
        """Initialize the trace analyzer.

        Args:
            trace_file_path: Path to the JSON-lines trace file.
        """
        self.trace_file_path = trace_file_path
        self._timeline: List[Dict[str, Any]] = []

    def load(self) -> List[Dict[str, Any]]:
        """Load and return the trace timeline.

        Returns:
            List of event dictionaries in chronological order.
        """
        if not os.path.exists(self.trace_file_path):
            return []

        events = []
        with open(self.trace_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse trace line: {line}")
        self._timeline = sorted(events, key=lambda e: e.get('timestamp_ms', 0))
        return self._timeline

    def get_stage_durations(self) -> Dict[str, Dict[str, Any]]:
        """Calculate detailed duration metrics for each stage.

        Returns:
            Dictionary mapping stage names to duration stats.
        """
        if not self._timeline:
            self.load()

        stage_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'runs': [],
            'start_times': [],
            'end_times': [],
        })

        for event in self._timeline:
            stage_name = event.get('stage_name', '')
            event_type = event.get('event_type', '')
            timestamp_ms = event.get('timestamp_ms', 0)

            if event_type in ('before_stage', EVENT_STAGE_TRACE_START):
                stage_data[stage_name]['start_times'].append(timestamp_ms)
            elif event_type in ('after_stage', EVENT_STAGE_TRACE_END):
                stage_data[stage_name]['end_times'].append(timestamp_ms)

        # Calculate durations
        result = {}
        for stage_name, data in stage_data.items():
            starts = data['start_times']
            ends = data['end_times']

            durations = []
            for start, end in zip(starts, ends):
                durations.append(end - start)

            if durations:
                result[stage_name] = {
                    'count': len(durations),
                    'total_ms': sum(durations),
                    'avg_ms': sum(durations) / len(durations),
                    'min_ms': min(durations),
                    'max_ms': max(durations),
                }

        return result

    def get_resource_timeline(self) -> List[Dict[str, Any]]:
        """Extract memory and resource usage over time from trace events.

        Returns:
            List of events containing memory information.
        """
        if not self._timeline:
            self.load()

        resource_events = []
        for event in self._timeline:
            metadata = event.get('metadata_dict', {})
            if 'memory_rss_mb' in metadata or 'memory_percent' in metadata:
                resource_events.append({
                    'timestamp_ms': event.get('timestamp_ms'),
                    'stage_name': event.get('stage_name'),
                    'memory_rss_mb': metadata.get('memory_rss_mb', 0),
                    'memory_percent': metadata.get('memory_percent', 0),
                })

        return resource_events

    def get_checkpoint_state_timeline(self) -> List[Dict[str, Any]]:
        """Extract checkpoint state changes over time.

        Returns:
            List of events showing checkpoint state at each stage.
        """
        if not self._timeline:
            self.load()

        checkpoint_events = []
        for event in self._timeline:
            metadata = event.get('metadata_dict', {})
            if 'checkpoint_state' in metadata or 'match_count' in metadata:
                checkpoint_events.append({
                    'timestamp_ms': event.get('timestamp_ms'),
                    'stage_name': event.get('stage_name'),
                    'event_type': event.get('event_type'),
                    'checkpoint_state': metadata.get('checkpoint_state', {}),
                    'match_count': metadata.get('match_count', 0),
                })

        return checkpoint_events

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive trace analysis report.

        Returns:
            Dictionary containing:
            - stage_durations: Detailed duration metrics per stage
            - total_duration_ms: Total pipeline duration
            - resource_timeline: Memory usage over time
            - checkpoint_timeline: Checkpoint state changes
            - stage_summary: Ordered list of stages with timing
        """
        if not self._timeline:
            self.load()

        # Get stage durations
        stage_durations = self.get_stage_durations()

        # Calculate total duration
        total_duration_ms = 0
        if self._timeline:
            first_ts = self._timeline[0].get('timestamp_ms', 0)
            last_ts = self._timeline[-1].get('timestamp_ms', 0)
            total_duration_ms = last_ts - first_ts

        # Get resource and checkpoint timelines
        resource_timeline = self.get_resource_timeline()
        checkpoint_timeline = self.get_checkpoint_state_timeline()

        # Build stage summary (ordered by first start time)
        stage_order = []
        stage_first_start: Dict[str, int] = {}
        for event in self._timeline:
            stage_name = event.get('stage_name', '')
            ts = event.get('timestamp_ms', 0)
            if stage_name and event.get('event_type') in ('before_stage', EVENT_STAGE_TRACE_START):
                if stage_name not in stage_first_start:
                    stage_first_start[stage_name] = ts

        for stage_name in sorted(stage_first_start.keys(), key=lambda s: stage_first_start[s]):
            stage_order.append({
                'stage': stage_name,
                'first_start_ms': stage_first_start[stage_name],
                'duration_ms': stage_durations.get(stage_name, {}).get('total_ms', 0),
            })

        return {
            'stage_durations': stage_durations,
            'total_duration_ms': total_duration_ms,
            'resource_timeline': resource_timeline,
            'checkpoint_timeline': checkpoint_timeline,
            'stage_summary': stage_order,
        }


class VerboseTraceHandler:
    """Handler that outputs detailed trace information to console/logs.

    Usage:
        handler = VerboseTraceHandler()
        bus.subscribe('before_stage', handler.on_trace_event)
        bus.subscribe('after_stage', handler.on_trace_event)
    """

    def __init__(self, include_memory: bool = True):
        """Initialize the verbose trace handler.

        Args:
            include_memory: Whether to include memory usage in trace output.
        """
        self.include_memory = include_memory
        self._stage_starts: Dict[str, float] = {}

    def on_trace_event(self, event: PipelineEvent) -> None:
        """Handle trace events and output detailed information.

        Args:
            event: The pipeline event to trace.
        """
        if event.event_type == 'before_stage':
            self._stage_starts[event.stage_name] = event.timestamp
            memory_info = ""
            if self.include_memory:
                mem = get_memory_usage()
                memory_info = f" | Memory: {mem.get('rss_mb', 0):.1f}MB"

            logger.info(
                f"[TRACE] Stage START: {event.stage_name}{memory_info}"
            )

        elif event.event_type == 'after_stage':
            start_time = self._stage_starts.pop(event.stage_name, event.timestamp)
            duration = event.timestamp - start_time

            # Extract useful context from event data
            context_parts = []
            if 'match_count' in event.data:
                context_parts.append(f"matches={event.data['match_count']}")
            if 'success' in event.data:
                context_parts.append(f"success={event.data['success']}")
            if 'elapsed' in event.data:
                context_parts.append(f"elapsed={event.data['elapsed']:.2f}s")

            context_str = f" | {', '.join(context_parts)}" if context_parts else ""

            status = "OK" if event.data.get('success') else "FAILED"
            logger.info(
                f"[TRACE] Stage END: {event.stage_name} | Duration: {duration:.2f}s | {status}{context_str}"
            )


# US-125-010: Webhook notification handler
class WebhookSender:
    """Webhook sender for pipeline event notifications (US-125-010).

    Sends HTTP POST notifications to configurable URLs when pipeline events
    occur. Supports different webhook URLs per event type, custom headers,
    and retry logic for failed deliveries.

    Usage:
        from src.config.sections.infrastructure import WebhookConfig

        config = WebhookConfig(
            enabled=True,
            default_url="https://example.com/webhook",
            event_urls={
                "after_stage": "https://example.com/webhook/stage",
                "on_pipeline_complete": "https://example.com/webhook/complete"
            },
            headers={"Authorization": "Bearer token"},
            max_retries=3
        )

        sender = WebhookSender(config)
        bus.subscribe('after_stage', sender.on_event)
        bus.subscribe('on_pipeline_complete', sender.on_event)
    """

    def __init__(self, webhook_config: Any):
        """Initialize the webhook sender.

        Args:
            webhook_config: WebhookConfig instance with webhook settings.
        """
        self.config = webhook_config
        self._retry_counts: Dict[str, int] = {}  # Track retries per URL

    def on_event(self, event: PipelineEvent) -> None:
        """Handle pipeline events and send webhook notifications.

        Args:
            event: The PipelineEvent to send as webhook notification.
        """
        # Skip if webhooks disabled
        if not self.config.enabled:
            return

        # US-138-012: Apply event and stage filtering
        if not self.config.should_send_event(event.event_type, event.stage_name):
            logger.debug(
                f"Webhook filtered: event_type={event.event_type}, stage_name={event.stage_name}"
            )
            return

        # Get URL for this event type
        url = self.config.get_url_for_event(event.event_type)
        if not url:
            logger.debug(f"No webhook URL configured for event type: {event.event_type}")
            return

        # Build payload
        payload = self._build_payload(event)

        # Send with retry
        self._send_with_retry(url, payload, event.event_type)

    def _build_payload(self, event: PipelineEvent) -> Dict[str, Any]:
        """Build webhook payload from pipeline event.

        Args:
            event: The PipelineEvent to convert.

        Returns:
            Dictionary payload for webhook JSON body.
        """
        import datetime

        payload = {
            "event_type": event.event_type,
            "stage_name": event.stage_name,
            "timestamp": datetime.datetime.fromtimestamp(event.timestamp).isoformat(),
            "timestamp_unix": event.timestamp,
        }

        # Include full event data if configured
        if self.config.include_full_data:
            payload["data"] = event.data
        else:
            # Include only key fields
            if "match_count" in event.data:
                payload["match_count"] = event.data["match_count"]
            if "success" in event.data:
                payload["success"] = event.data["success"]
            if "elapsed" in event.data:
                payload["elapsed"] = event.data["elapsed"]

        # Include errors if configured
        if self.config.include_errors and event.error:
            payload["error"] = event.error

        return payload

    def _send_with_retry(self, url: str, payload: Dict[str, Any], event_type: str) -> bool:
        """Send webhook with exponential backoff retry logic.

        Args:
            url: The webhook URL to send to.
            payload: The JSON payload to send.
            event_type: The event type for retry tracking.

        Returns:
            True if delivery succeeded, False otherwise.
        """
        # Get retry count for this URL
        retry_key = f"{url}:{event_type}"
        max_retries = self.config.max_retries
        base_delay = self.config.retry_base_delay
        max_delay = self.config.retry_max_delay
        timeout = self.config.timeout_seconds

        for attempt in range(max_retries + 1):
            try:
                # Build request
                data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={
                        "Content-Type": "application/json",
                        **self.config.headers,
                    },
                    method="POST"
                )

                # Execute request with timeout
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    if response.status < 400:
                        logger.info(
                            f"Webhook delivered successfully: {event_type} to {url} "
                            f"(attempt {attempt + 1}/{max_retries + 1})"
                        )
                        # Reset retry count on success
                        self._retry_counts.pop(retry_key, None)
                        return True
                    else:
                        logger.warning(
                            f"Webhook returned HTTP {response.status}: {event_type} to {url}"
                        )

            except urllib.error.HTTPError as e:
                logger.warning(
                    f"Webhook HTTP error {e.code}: {event_type} to {url} "
                    f"(attempt {attempt + 1}/{max_retries + 1})"
                )
            except urllib.error.URLError as e:
                logger.warning(
                    f"Webhook URL error: {event_type} to {url} - {e.reason} "
                    f"(attempt {attempt + 1}/{max_retries + 1})"
                )
            except TimeoutError:
                logger.warning(
                    f"Webhook timeout: {event_type} to {url} "
                    f"(attempt {attempt + 1}/{max_retries + 1})"
                )
            except Exception as e:
                logger.warning(
                    f"Webhook delivery failed: {event_type} to {url} - {type(e).__name__}: {e} "
                    f"(attempt {attempt + 1}/{max_retries + 1})"
                )

            # Check if we should retry
            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = min(base_delay * (2 ** attempt), max_delay)
                delay = delay * (0.5 + random.random())  # Add jitter
                logger.debug(f"Retrying webhook in {delay:.2f}s: {event_type} to {url}")
                time.sleep(delay)

        # All retries exhausted
        logger.error(
            f"Webhook delivery failed after {max_retries + 1} attempts: "
            f"{event_type} to {url}"
        )
        return False

    def _send(self, url: str, payload: Dict[str, Any]) -> bool:
        """Send a single webhook request (no retry).

        Args:
            url: The webhook URL to send to.
            payload: The JSON payload to send.

        Returns:
            True if delivery succeeded, False otherwise.
        """
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    **self.config.headers,
                },
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                return response.status < 400

        except Exception as e:
            logger.warning(f"Webhook delivery failed: {url} - {type(e).__name__}: {e}")
            return False
