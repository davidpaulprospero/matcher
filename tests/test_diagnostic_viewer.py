#!/usr/bin/env python3
"""Tests for scripts/diagnostic_viewer.py"""

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import pytest

# Add scripts directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.diagnostic_viewer import (
    parse_timestamp,
    parse_log_line,
    load_json_log,
    load_plain_log,
    load_logs,
    filter_entries,
    display_entries,
    save_entries,
)


class TestParseTimestamp:
    """Test timestamp parsing."""

    def test_parse_standard_format(self):
        """Test standard YYYY-MM-DD HH:MM:SS format."""
        result = parse_timestamp("2026-02-17 10:30:45")
        assert result is not None
        assert result.year == 2026
        assert result.month == 2
        assert result.day == 17
        assert result.hour == 10
        assert result.minute == 30
        assert result.second == 45

    def test_parse_iso_format(self):
        """Test ISO format YYYY-MM-DDTHH:MM:SS."""
        result = parse_timestamp("2026-02-17T10:30:45")
        assert result is not None
        assert result.year == 2026
        assert result.month == 2
        assert result.day == 17
        assert result.hour == 10
        assert result.minute == 30

    def test_parse_iso_format_with_microseconds(self):
        """Test ISO format with microseconds."""
        result = parse_timestamp("2026-02-17T10:30:45.123456")
        assert result is not None
        assert result.year == 2026
        assert result.month == 2
        assert result.day == 17
        assert result.hour == 10
        assert result.minute == 30
        assert result.second == 45

    def test_parse_time_only(self):
        """Test time-only format HH:MM:SS."""
        result = parse_timestamp("10:30:45")
        assert result is not None
        assert result.hour == 10
        assert result.minute == 30
        assert result.second == 45

    def test_parse_invalid_format(self):
        """Test invalid timestamp returns None."""
        result = parse_timestamp("not-a-timestamp")
        assert result is None

    def test_parse_empty_string(self):
        """Test empty string returns None."""
        result = parse_timestamp("")
        assert result is None


class TestParseLogLine:
    """Test plain text log line parsing."""

    def test_parse_valid_line(self):
        """Test parsing a valid log line."""
        line = "2026-02-17 10:30:45 | ERROR | Something went wrong"
        result = parse_log_line(line)
        assert result is not None
        assert result['timestamp'] == "2026-02-17 10:30:45"
        assert result['level'] == "ERROR"
        assert result['message'] == "Something went wrong"
        assert result['source'] == 'log'

    def test_parse_line_without_message(self):
        """Test parsing line with empty message returns None (regex requires .+)."""
        line = "2026-02-17 10:30:45 | INFO | "
        result = parse_log_line(line)
        # Empty message after pipe returns None (regex requires .+)
        assert result is None

    def test_parse_invalid_line(self):
        """Test parsing invalid line returns None."""
        line = "This is not a valid log line"
        result = parse_log_line(line)
        assert result is None

    def test_parse_line_without_pipe_separator(self):
        """Test parsing line without pipe separator."""
        line = "2026-02-17 10:30:45 ERROR Something went wrong"
        result = parse_log_line(line)
        assert result is None

    def test_parse_line_different_levels(self):
        """Test parsing lines with different log levels."""
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            line = f"2026-02-17 10:30:45 | {level} | Test message"
            result = parse_log_line(line)
            assert result is not None
            assert result['level'] == level


class TestLoadJsonLog:
    """Test JSON log file loading."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_load_basic_json_log(self, temp_log_dir):
        """Test loading basic JSON log file."""
        log_file = temp_log_dir / "test_run.json"
        log_data = {
            "run_id": "test-run-001",
            "start_time": "2026-02-17 10:00:00",
            "errors": ["Error 1", "Error 2"],
            "warnings": ["Warning 1"],
        }
        with open(log_file, 'w') as f:
            json.dump(log_data, f)

        entries = load_json_log(log_file)
        assert len(entries) == 3  # 2 errors + 1 warning

        error_entries = [e for e in entries if e['level'] == 'ERROR']
        warning_entries = [e for e in entries if e['level'] == 'WARNING']
        assert len(error_entries) == 2
        assert len(warning_entries) == 1

    def test_load_json_with_api_calls(self, temp_log_dir):
        """Test loading JSON with failed API calls."""
        log_file = temp_log_dir / "test_api.json"
        log_data = {
            "run_id": "test-run-002",
            "start_time": "2026-02-17 11:00:00",
            "api_calls": [
                {"provider": "openai", "endpoint": "chat", "success": False, "error": "Rate limited", "timestamp": "2026-02-17 11:00:00"},
                {"provider": "anthropic", "endpoint": "messages", "success": True},
            ]
        }
        with open(log_file, 'w') as f:
            json.dump(log_data, f)

        entries = load_json_log(log_file)
        assert len(entries) == 1
        assert entries[0]['level'] == 'ERROR'
        assert 'Rate limited' in entries[0]['message']

    def test_load_json_with_stage_timings(self, temp_log_dir):
        """Test loading JSON with stage timings (requires errors/warnings/api_calls to trigger processing)."""
        log_file = temp_log_dir / "test_timing.json"
        log_data = {
            "run_id": "test-run-003",
            "start_time": "2026-02-17 12:00:00",
            "errors": [],  # Need errors/warnings/api_calls for processing
            "stage_timings": {
                "download": 120.5,
                "transcribe": 60.0,
                "match": 45.2,
            }
        }
        with open(log_file, 'w') as f:
            json.dump(log_data, f)

        entries = load_json_log(log_file)
        assert len(entries) == 3
        for entry in entries:
            assert entry['level'] == 'INFO'
            assert 'Stage' in entry['message']

    def test_load_json_with_summary(self, temp_log_dir):
        """Test loading JSON with summary (requires errors/warnings/api_calls to trigger processing)."""
        log_file = temp_log_dir / "test_summary.json"
        log_data = {
            "run_id": "test-run-004",
            "start_time": "2026-02-17 13:00:00",
            "errors": [],  # Need errors/warnings/api_calls for processing
            "summary": {
                "total_matches": 10,
                "total_segments": 20,
                "total_api_calls": 50,
                "total_api_cost_est_usd": 0.25,
            }
        }
        with open(log_file, 'w') as f:
            json.dump(log_data, f)

        entries = load_json_log(log_file)
        assert len(entries) == 1
        assert '10/20 matches' in entries[0]['message']

    def test_load_invalid_json(self, temp_log_dir):
        """Test loading invalid JSON returns empty list."""
        log_file = temp_log_dir / "invalid.json"
        with open(log_file, 'w') as f:
            f.write("not valid json")

        entries = load_json_log(log_file)
        assert entries == []


class TestLoadPlainLog:
    """Test plain text log file loading."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for log files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_load_plain_log_file(self, temp_log_dir):
        """Test loading plain text log file."""
        log_file = temp_log_dir / "test.log"
        with open(log_file, 'w') as f:
            f.write("2026-02-17 10:00:00 | INFO | First message\n")
            f.write("2026-02-17 10:01:00 | ERROR | Second message\n")
            f.write("2026-02-17 10:02:00 | WARNING | Third message\n")
            f.write("Invalid line\n")

        entries = load_plain_log(log_file)
        assert len(entries) == 3

    def test_load_empty_log_file(self, temp_log_dir):
        """Test loading empty log file."""
        log_file = temp_log_dir / "empty.log"
        with open(log_file, 'w') as f:
            pass

        entries = load_plain_log(log_file)
        assert entries == []


class TestFilterEntries:
    """Test log entry filtering."""

    def get_sample_entries(self) -> List[Dict[str, Any]]:
        """Create sample log entries for testing."""
        return [
            {'timestamp': '2026-02-17 10:00:00', 'level': 'DEBUG', 'message': 'Debug message'},
            {'timestamp': '2026-02-17 10:01:00', 'level': 'INFO', 'message': 'Info message'},
            {'timestamp': '2026-02-17 10:02:00', 'level': 'WARNING', 'message': 'Rate limit hit'},
            {'timestamp': '2026-02-17 10:03:00', 'level': 'ERROR', 'message': 'Network error occurred'},
            {'timestamp': '2026-02-17 10:04:00', 'level': 'ERROR', 'message': 'Download failed: 404'},
            {'timestamp': '2026-02-17 10:05:00', 'level': 'CRITICAL', 'message': 'Critical failure'},
        ]

    def test_filter_by_level_debug(self):
        """Test filtering by DEBUG level."""
        entries = self.get_sample_entries()
        result = filter_entries(entries, level='DEBUG')
        assert len(result) == 1
        assert result[0]['level'] == 'DEBUG'

    def test_filter_by_level_info(self):
        """Test filtering by INFO level."""
        entries = self.get_sample_entries()
        result = filter_entries(entries, level='INFO')
        assert len(result) == 1
        assert result[0]['level'] == 'INFO'

    def test_filter_by_level_error(self):
        """Test filtering by ERROR level."""
        entries = self.get_sample_entries()
        result = filter_entries(entries, level='ERROR')
        assert len(result) == 2
        for entry in result:
            assert entry['level'] == 'ERROR'

    def test_filter_by_level_case_insensitive(self):
        """Test level filtering is case insensitive."""
        entries = self.get_sample_entries()
        result = filter_entries(entries, level='error')
        assert len(result) == 2

    def test_filter_by_category_rate_limit(self):
        """Test filtering by rate_limit category."""
        entries = self.get_sample_entries()
        result = filter_entries(entries, category='rate_limit')
        assert len(result) == 1
        assert 'Rate limit' in result[0]['message']

    def test_filter_by_category_network(self):
        """Test filtering by network category."""
        entries = self.get_sample_entries()
        result = filter_entries(entries, category='network')
        assert len(result) == 1
        assert 'Network' in result[0]['message']

    def test_filter_by_category_download(self):
        """Test filtering by download category."""
        entries = self.get_sample_entries()
        result = filter_entries(entries, category='download')
        assert len(result) == 1
        assert 'Download' in result[0]['message']

    def test_filter_by_category_unknown(self):
        """Test filtering by unknown category."""
        entries = self.get_sample_entries()
        # Add unknown error entry
        entries.append({'timestamp': '2026-02-17 10:06:00', 'level': 'ERROR', 'message': 'Unknown error'})
        result = filter_entries(entries, category='unknown')
        assert len(result) == 1
        assert 'Unknown' in result[0]['message']

    def test_filter_by_from_time(self):
        """Test filtering by from_time."""
        entries = self.get_sample_entries()
        from_time = datetime(2026, 2, 17, 10, 2, 0)
        result = filter_entries(entries, from_time=from_time)
        assert len(result) == 4  # 10:02:00 and after

    def test_filter_by_to_time(self):
        """Test filtering by to_time."""
        entries = self.get_sample_entries()
        to_time = datetime(2026, 2, 17, 10, 2, 0)
        result = filter_entries(entries, to_time=to_time)
        assert len(result) == 3  # 10:00:00, 10:01:00, 10:02:00

    def test_filter_by_time_range(self):
        """Test filtering by both from_time and to_time."""
        entries = self.get_sample_entries()
        from_time = datetime(2026, 2, 17, 10, 1, 0)
        to_time = datetime(2026, 2, 17, 10, 3, 0)
        result = filter_entries(entries, from_time=from_time, to_time=to_time)
        assert len(result) == 3  # 10:01:00, 10:02:00, 10:03:00

    def test_filter_by_search_term(self):
        """Test filtering by search term."""
        entries = self.get_sample_entries()
        result = filter_entries(entries, search='error')
        # Only "Network error occurred" contains "error" (case-sensitive)
        assert len(result) == 1
        assert 'Network error' in result[0]['message']

    def test_filter_combined_level_and_category(self):
        """Test filtering by both level and category."""
        entries = self.get_sample_entries()
        result = filter_entries(entries, level='ERROR', category='network')
        assert len(result) == 1
        assert 'Network' in result[0]['message']

    def test_filter_no_match(self):
        """Test filtering with no matching entries."""
        entries = self.get_sample_entries()
        result = filter_entries(entries, level='DEBUG', category='auth')
        assert len(result) == 0

    def test_filter_no_criteria(self):
        """Test filtering with no criteria returns all."""
        entries = self.get_sample_entries()
        result = filter_entries(entries)
        assert len(result) == len(entries)


class TestSaveEntries:
    """Test saving log entries to files."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_save_to_json(self, temp_dir):
        """Test saving entries to JSON file."""
        entries = [
            {'timestamp': '2026-02-17 10:00:00', 'level': 'ERROR', 'message': 'Test error'},
            {'timestamp': '2026-02-17 10:01:00', 'level': 'INFO', 'message': 'Test info'},
        ]
        output_path = temp_dir / "output.json"

        result = save_entries(entries, str(output_path))
        assert result is True
        assert output_path.exists()

        # Verify JSON can be loaded
        with open(output_path) as f:
            data = json.load(f)
        assert 'entries' in data
        assert len(data['entries']) == 2
        assert 'metadata' in data
        assert data['metadata']['total_entries'] == 2

    def test_save_to_txt(self, temp_dir):
        """Test saving entries to text file."""
        entries = [
            {'timestamp': '2026-02-17 10:00:00', 'level': 'ERROR', 'message': 'Test error'},
            {'timestamp': '2026-02-17 10:01:00', 'level': 'INFO', 'message': 'Test info'},
        ]
        output_path = temp_dir / "output.txt"

        result = save_entries(entries, str(output_path))
        assert result is True
        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            content = f.read()
        assert 'ERROR' in content
        assert 'Test error' in content
        assert 'INFO' in content

    def test_save_to_txt_with_run_id(self, temp_dir):
        """Test saving entries with run_id to text file."""
        entries = [
            {'timestamp': '2026-02-17 10:00:00', 'level': 'ERROR', 'message': 'Test error', 'run_id': 'run-001'},
        ]
        output_path = temp_dir / "output.txt"

        result = save_entries(entries, str(output_path))
        assert result is True

        with open(output_path) as f:
            content = f.read()
        assert 'run: run-001' in content

    def test_save_empty_entries(self, temp_dir):
        """Test saving empty entries returns False."""
        entries = []
        output_path = temp_dir / "output.json"

        result = save_entries(entries, str(output_path))
        assert result is False

    def test_save_unsupported_format(self, temp_dir):
        """Test saving to unsupported format returns False."""
        entries = [{'timestamp': '2026-02-17 10:00:00', 'level': 'INFO', 'message': 'Test'}]
        output_path = temp_dir / "output.csv"

        result = save_entries(entries, str(output_path))
        assert result is False


class TestLoadLogs:
    """Test loading logs from directory."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory with sample logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            # Create JSON log
            json_log = log_dir / "run1.json"
            with open(json_log, 'w') as f:
                json.dump({"run_id": "r1", "errors": ["e1"]}, f)

            # Create plain log
            plain_log = log_dir / "test.log"
            with open(plain_log, 'w') as f:
                f.write("2026-02-17 10:00:00 | INFO | Log entry\n")

            yield log_dir

    def test_load_all_logs(self, temp_log_dir):
        """Test loading all log files."""
        entries = load_logs(temp_log_dir)
        # 1 error from JSON + 1 info from plain log = 2
        assert len(entries) == 2

    def test_load_json_only(self, temp_log_dir):
        """Test loading only JSON files."""
        entries = load_logs(temp_log_dir, json_only=True)
        assert len(entries) == 1
        assert entries[0]['level'] == 'ERROR'

    def test_load_nonexistent_dir(self):
        """Test loading from nonexistent directory returns empty."""
        entries = load_logs(Path("/nonexistent/directory"))
        assert entries == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
