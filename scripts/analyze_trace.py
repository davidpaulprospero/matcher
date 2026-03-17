#!/usr/bin/env python3
"""
Pipeline Trace Analysis Script (US-108-007)

Analyzes pipeline_trace.jsonl to identify:
- Longest stages
- Most common errors
- Retry patterns

Usage:
    python scripts/analyze_trace.py /path/to/project/pipeline_trace.jsonl
    python scripts/analyze_trace.py /path/to/project/pipeline_trace.jsonl --json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

# Import standardized output functions
from script_utils import print_error, print_info, print_header

# Change to project root so relative paths work correctly
os.chdir(project_root)


def load_trace(trace_path: str) -> list[dict]:
    """Load events from JSON-lines trace file."""
    if not os.path.exists(trace_path):
        print_error(f"Trace file not found: {trace_path}")
        sys.exit(1)

    events: list[dict] = []
    with open(trace_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # Skip invalid lines

    return sorted(events, key=lambda e: e.get('timestamp_ms', 0))


def analyze_trace(events: list[dict]) -> dict:
    """Analyze trace events to identify patterns."""
    if not events:
        return {
            'longest_stages': [],
            'most_common_errors': [],
            'retry_patterns': [],
            'summary': {'total_events': 0},
        }

    # Find longest stages
    stage_durations: defaultdict = defaultdict(list)
    stage_start: dict = {}
    for event in events:
        event_type: str = event.get('event_type', '')
        stage_name: str = event.get('stage_name', '')
        timestamp_ms: int = event.get('timestamp_ms', 0)

        if event_type == 'before_stage':
            stage_start[stage_name] = timestamp_ms
        elif event_type == 'after_stage' and stage_name in stage_start:
            duration: int = timestamp_ms - stage_start[stage_name]
            stage_durations[stage_name].append(duration)
            del stage_start[stage_name]

    longest_stages: list[dict] = [
        {
            'stage': stage,
            'total_duration_ms': sum(durations),
            'count': len(durations),
            'avg_duration_ms': sum(durations) // len(durations) if durations else 0,
        }
        for stage, durations in stage_durations.items()
    ]
    longest_stages.sort(key=lambda x: x['total_duration_ms'], reverse=True)

    # Find most common errors
    error_counts: defaultdict = defaultdict(int)
    for event in events:
        if event.get('error'):
            error_counts[event['error']] += 1

    most_common_errors: list[dict] = [
        {'error': error, 'count': count}
        for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    # Find retry patterns
    error_recovered_count: int = sum(
        1 for e in events if e.get('event_type') == 'error_recovered'
    )

    # Find stage retries (before_stage followed by another before_stage for same stage)
    stage_retries: defaultdict = defaultdict(int)
    last_stage_start: dict = {}
    for event in events:
        event_type: str = event.get('event_type', '')
        stage_name: str = event.get('stage_name', '')
        timestamp_ms: int = event.get('timestamp_ms', 0)

        if event_type == 'before_stage':
            if stage_name in last_stage_start:
                stage_retries[stage_name] += 1
            last_stage_start[stage_name] = timestamp_ms

    retry_patterns: list[dict] = [
        {'type': 'error_recovered', 'count': error_recovered_count},
        {'type': 'stage_retries', 'details': dict(stage_retries)},
    ]

    # Summary stats
    event_types: defaultdict = defaultdict(int)
    for event in events:
        event_types[event.get('event_type', 'unknown')] += 1

    first_ts: int = events[0].get('timestamp_ms', 0)
    last_ts: int = events[-1].get('timestamp_ms', 0)

    summary: dict = {
        'total_events': len(events),
        'pipeline_duration_ms': last_ts - first_ts,
        'event_types': dict(event_types),
    }

    return {
        'longest_stages': longest_stages,
        'most_common_errors': most_common_errors,
        'retry_patterns': retry_patterns,
        'summary': summary,
    }


def format_human(analysis: dict) -> str:
    """Format analysis for human-readable output."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("PIPELINE TRACE ANALYSIS")
    lines.append("=" * 60)

    summary: dict = analysis.get('summary', {})
    lines.append(f"\nSummary:")
    lines.append(f"  Total Events: {summary.get('total_events', 0)}")
    lines.append(f"  Pipeline Duration: {summary.get('pipeline_duration_ms', 0) / 1000:.2f}s")

    event_types: dict = summary.get('event_types', {})
    if event_types:
        lines.append(f"  Event Types: {', '.join(f'{k}={v}' for k, v in event_types.items())}")

    longest: list = analysis.get('longest_stages', [])
    if longest:
        lines.append(f"\nLongest Stages:")
        for i, stage in enumerate(longest[:5], 1):
            lines.append(
                f"  {i}. {stage['stage']}: {stage['total_duration_ms'] / 1000:.2f}s "
                f"(x{stage['count']}, avg: {stage['avg_duration_ms'] / 1000:.2f}s)"
            )

    errors: list = analysis.get('most_common_errors', [])
    if errors:
        lines.append(f"\nMost Common Errors:")
        for i, err in enumerate(errors[:5], 1):
            lines.append(f"  {i}. {err['error']} (x{err['count']})")

    retries: list = analysis.get('retry_patterns', [])
    if retries:
        lines.append(f"\nRetry Patterns:")
        for pattern in retries:
            if pattern['type'] == 'error_recovered':
                lines.append(f"  - Error Recoveries: {pattern['count']}")
            elif pattern['type'] == 'stage_retries' and pattern['details']:
                for stage, count in pattern['details'].items():
                    if count > 0:
                        lines.append(f"  - {stage} retries: {count}")

    lines.append("\n" + "=" * 60)
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pipeline_trace.jsonl to identify patterns"
    )
    parser.add_argument(
        'trace_file',
        nargs='?',
        help='Path to pipeline_trace.jsonl'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON instead of human-readable'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Include verbose output'
    )

    args = parser.parse_args()

    # If no trace file provided, look for it in common locations
    trace_file = args.trace_file
    if not trace_file:
        # Check current directory and parent
        for path in ['pipeline_trace.jsonl', '../pipeline_trace.jsonl']:
            if os.path.exists(path):
                trace_file = path
                break

    if not trace_file:
        print_error("No trace file specified and none found in current directory")
        parser.print_help()
        sys.exit(1)

    events = load_trace(trace_file)
    analysis = analyze_trace(events)

    if args.json:
        print_info(json.dumps(analysis, indent=2))
    else:
        print_header("Trace Analysis Results")
        print(format_human(analysis))

    # Exit with error code if there were errors
    if analysis.get('most_common_errors'):
        sys.exit(1)


if __name__ == '__main__':
    main()
