#!/usr/bin/env python3
"""Tests for scripts/analyze_trace.py"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.analyze_trace import (
    load_trace,
    analyze_trace,
    format_human,
)


class TestLoadTrace:
    """Test trace file loading."""

    def test_load_trace_valid_jsonl(self):
        """Test loading a valid JSON-lines trace file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"event_type": "before_stage", "stage_name": "VIDEO_SEARCH", "timestamp_ms": 1000}) + "\n")
            f.write(json.dumps({"event_type": "after_stage", "stage_name": "VIDEO_SEARCH", "timestamp_ms": 5000}) + "\n")
            f.write(json.dumps({"event_type": "error", "error": "Test error", "timestamp_ms": 3000}) + "\n")
            temp_path = f.name

        try:
            events = load_trace(temp_path)
            assert len(events) == 3
            assert events[0]["timestamp_ms"] == 1000
            assert events[1]["timestamp_ms"] == 3000
            assert events[2]["timestamp_ms"] == 5000
        finally:
            os.unlink(temp_path)

    def test_load_trace_file_not_found(self, capsys):
        """Test loading non-existent file exits with error."""
        with pytest.raises(SystemExit) as exc_info:
            load_trace("/nonexistent/path/trace.jsonl")
        assert exc_info.value.code == 1

    def test_load_trace_skips_invalid_lines(self):
        """Test that invalid JSON lines are skipped."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"event_type": "before_stage", "timestamp_ms": 1000}) + "\n")
            f.write("invalid json line\n")
            f.write(json.dumps({"event_type": "after_stage", "timestamp_ms": 2000}) + "\n")
            temp_path = f.name

        try:
            events = load_trace(temp_path)
            assert len(events) == 2
        finally:
            os.unlink(temp_path)

    def test_load_trace_empty_file(self):
        """Test loading an empty file returns empty list."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name

        try:
            events = load_trace(temp_path)
            assert events == []
        finally:
            os.unlink(temp_path)

    def test_load_trace_sorts_by_timestamp(self):
        """Test that events are sorted by timestamp_ms."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(json.dumps({"timestamp_ms": 5000}) + "\n")
            f.write(json.dumps({"timestamp_ms": 1000}) + "\n")
            f.write(json.dumps({"timestamp_ms": 3000}) + "\n")
            temp_path = f.name

        try:
            events = load_trace(temp_path)
            assert [e["timestamp_ms"] for e in events] == [1000, 3000, 5000]
        finally:
            os.unlink(temp_path)


class TestAnalyzeTrace:
    """Test trace analysis."""

    def test_analyze_empty_events(self):
        """Test analyze with empty events list returns default structure."""
        result = analyze_trace([])
        assert result['longest_stages'] == []
        assert result['most_common_errors'] == []
        assert result['retry_patterns'] == []
        assert result['summary']['total_events'] == 0

    def test_analyze_identifies_longest_stages(self):
        """Test that analyze correctly identifies longest stages."""
        events = [
            {"event_type": "before_stage", "stage_name": "VIDEO_SEARCH", "timestamp_ms": 0},
            {"event_type": "after_stage", "stage_name": "VIDEO_SEARCH", "timestamp_ms": 10000},
            {"event_type": "before_stage", "stage_name": "DOWNLOAD", "timestamp_ms": 10000},
            {"event_type": "after_stage", "stage_name": "DOWNLOAD", "timestamp_ms": 15000},
            {"event_type": "before_stage", "stage_name": "MATCH", "timestamp_ms": 15000},
            {"event_type": "after_stage", "stage_name": "MATCH", "timestamp_ms": 30000},
        ]

        result = analyze_trace(events)

        assert len(result['longest_stages']) == 3
        # Should be sorted by total_duration_ms descending
        # MATCH: 15000ms (15000-30000), VIDEO_SEARCH: 10000ms (0-10000), DOWNLOAD: 5000ms (10000-15000)
        assert result['longest_stages'][0]['stage'] == "MATCH"
        assert result['longest_stages'][0]['total_duration_ms'] == 15000
        assert result['longest_stages'][1]['stage'] == "VIDEO_SEARCH"
        assert result['longest_stages'][2]['stage'] == "DOWNLOAD"

    def test_analyze_identifies_most_common_errors(self):
        """Test that analyze correctly identifies most common errors."""
        events = [
            {"event_type": "error", "error": "Rate limit", "timestamp_ms": 1000},
            {"event_type": "error", "error": "Rate limit", "timestamp_ms": 2000},
            {"event_type": "error", "error": "Rate limit", "timestamp_ms": 3000},
            {"event_type": "error", "error": "Network error", "timestamp_ms": 4000},
            {"event_type": "error", "error": "Network error", "timestamp_ms": 5000},
            {"event_type": "error", "error": "Timeout", "timestamp_ms": 6000},
        ]

        result = analyze_trace(events)

        assert len(result['most_common_errors']) == 3
        assert result['most_common_errors'][0]['error'] == "Rate limit"
        assert result['most_common_errors'][0]['count'] == 3
        assert result['most_common_errors'][1]['error'] == "Network error"
        assert result['most_common_errors'][1]['count'] == 2
        assert result['most_common_errors'][2]['error'] == "Timeout"
        assert result['most_common_errors'][2]['count'] == 1

    def test_analyze_retry_patterns_error_recovered(self):
        """Test detection of error_recovered events."""
        events = [
            {"event_type": "error", "error": "Test error", "timestamp_ms": 1000},
            {"event_type": "error_recovered", "timestamp_ms": 2000},
            {"event_type": "error_recovered", "timestamp_ms": 3000},
        ]

        result = analyze_trace(events)

        retry_patterns = result['retry_patterns']
        error_recovered = next(p for p in retry_patterns if p['type'] == 'error_recovered')
        assert error_recovered['count'] == 2

    def test_analyze_retry_patterns_stage_retries(self):
        """Test detection of stage retries (before_stage without after_stage)."""
        events = [
            {"event_type": "before_stage", "stage_name": "VIDEO_SEARCH", "timestamp_ms": 0},
            {"event_type": "before_stage", "stage_name": "VIDEO_SEARCH", "timestamp_ms": 5000},
            {"event_type": "after_stage", "stage_name": "VIDEO_SEARCH", "timestamp_ms": 10000},
        ]

        result = analyze_trace(events)

        retry_patterns = result['retry_patterns']
        stage_retries = next(p for p in retry_patterns if p['type'] == 'stage_retries')
        assert stage_retries['details']['VIDEO_SEARCH'] == 1

    def test_analyze_summary_stats(self):
        """Test summary statistics are calculated correctly."""
        events = [
            {"event_type": "before_stage", "timestamp_ms": 0},
            {"event_type": "after_stage", "timestamp_ms": 5000},
            {"event_type": "info", "timestamp_ms": 1000},
            {"event_type": "info", "timestamp_ms": 2000},
            {"event_type": "info", "timestamp_ms": 3000},
        ]

        result = analyze_trace(events)

        summary = result['summary']
        assert summary['total_events'] == 5
        # pipeline_duration_ms is last_ts - first_ts from sorted events
        # Events sorted: 0, 1000, 2000, 3000, 5000 -> last_ts=3000, first_ts=0 -> 3000-0=3000
        assert summary['pipeline_duration_ms'] == 3000
        assert summary['event_types']['before_stage'] == 1
        assert summary['event_types']['after_stage'] == 1
        assert summary['event_types']['info'] == 3

    def test_analyze_handles_missing_fields(self):
        """Test analysis handles events with missing optional fields."""
        events = [
            {"timestamp_ms": 1000},
            {"event_type": "before_stage", "stage_name": "TEST", "timestamp_ms": 2000},
            {"event_type": "after_stage", "stage_name": "TEST", "timestamp_ms": 3000},
        ]

        result = analyze_trace(events)

        assert result['summary']['total_events'] == 3
        assert len(result['longest_stages']) == 1
        assert result['longest_stages'][0]['stage'] == "TEST"
        assert result['longest_stages'][0]['total_duration_ms'] == 1000


class TestFormatHuman:
    """Test human-readable output formatting."""

    def test_format_empty_analysis(self):
        """Test formatting empty analysis results."""
        analysis = {
            'longest_stages': [],
            'most_common_errors': [],
            'retry_patterns': [],
            'summary': {'total_events': 0, 'pipeline_duration_ms': 0, 'event_types': {}},
        }

        output = format_human(analysis)
        assert "Summary:" in output
        assert "Total Events: 0" in output
        assert "PIPELINE TRACE ANALYSIS" in output

    def test_format_with_stages(self):
        """Test formatting with stage data."""
        analysis = {
            'longest_stages': [
                {'stage': 'VIDEO_SEARCH', 'total_duration_ms': 10000, 'count': 1, 'avg_duration_ms': 10000},
            ],
            'most_common_errors': [],
            'retry_patterns': [],
            'summary': {
                'total_events': 2,
                'pipeline_duration_ms': 10000,
                'event_types': {'before_stage': 1, 'after_stage': 1},
            },
        }

        output = format_human(analysis)
        assert "Longest Stages:" in output
        assert "VIDEO_SEARCH" in output
        assert "10.00s" in output

    def test_format_with_errors(self):
        """Test formatting with error data."""
        analysis = {
            'longest_stages': [],
            'most_common_errors': [
                {'error': 'Rate limit', 'count': 5},
                {'error': 'Network error', 'count': 2},
            ],
            'retry_patterns': [],
            'summary': {'total_events': 7, 'pipeline_duration_ms': 7000, 'event_types': {}},
        }

        output = format_human(analysis)
        assert "Most Common Errors:" in output
        assert "Rate limit" in output
        assert "x5" in output

    def test_format_with_retry_patterns(self):
        """Test formatting with retry patterns."""
        analysis = {
            'longest_stages': [],
            'most_common_errors': [],
            'retry_patterns': [
                {'type': 'error_recovered', 'count': 3},
                {'type': 'stage_retries', 'details': {'VIDEO_SEARCH': 2, 'DOWNLOAD': 1}},
            ],
            'summary': {'total_events': 10, 'pipeline_duration_ms': 10000, 'event_types': {}},
        }

        output = format_human(analysis)
        assert "Retry Patterns:" in output
        assert "Error Recoveries: 3" in output
        assert "VIDEO_SEARCH retries: 2" in output
