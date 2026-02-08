"""
Tests for TranscriptCache entry validation and integrity checks.

US-79-009: Cache warmup validation with integrity check.
"""

import json
import logging
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription.cache import TranscriptCache

pytestmark = [pytest.mark.fast, pytest.mark.unit]


@pytest.fixture
def cache_base(tmp_path):
    """Return a tmp_path to use as the base cache directory."""
    return tmp_path


def _write_cache_entry(directory: Path, filename: str, segments: list, source_file: str = "video.mp4"):
    """Helper: write a cache JSON file with given segments."""
    directory.mkdir(parents=True, exist_ok=True)
    # Add source_file to each segment so the cache can build its source map
    for seg in segments:
        if 'source_file' not in seg:
            seg['source_file'] = source_file
    path = directory / filename
    path.write_text(json.dumps(segments), encoding="utf-8")
    return path


class TestValidateEntry:
    """Unit tests for validate_entry() method."""

    def test_valid_entry_passes(self, cache_base):
        cache = TranscriptCache(str(cache_base))
        segments = [
            {'start': 0.0, 'end': 5.0, 'text': 'hello world'},
            {'start': 5.0, 'end': 10.0, 'text': 'goodbye'},
        ]
        assert cache.validate_entry(segments) is None

    def test_empty_segments_fails(self, cache_base):
        cache = TranscriptCache(str(cache_base))
        reason = cache.validate_entry([])
        assert reason is not None
        assert "empty segments" in reason

    def test_segment_missing_text_fails(self, cache_base):
        cache = TranscriptCache(str(cache_base))
        segments = [{'start': 0.0, 'end': 5.0, 'text': ''}]
        reason = cache.validate_entry(segments)
        assert reason is not None
        assert "empty text" in reason

    def test_segment_whitespace_only_text_fails(self, cache_base):
        cache = TranscriptCache(str(cache_base))
        segments = [{'start': 0.0, 'end': 5.0, 'text': '   '}]
        reason = cache.validate_entry(segments)
        assert reason is not None
        assert "empty text" in reason

    def test_segment_missing_times_fails(self, cache_base):
        cache = TranscriptCache(str(cache_base))
        segments = [{'text': 'hello'}]
        reason = cache.validate_entry(segments)
        assert reason is not None
        assert "missing" in reason

    def test_segment_end_before_start_fails(self, cache_base):
        cache = TranscriptCache(str(cache_base))
        segments = [{'start': 10.0, 'end': 5.0, 'text': 'hello'}]
        reason = cache.validate_entry(segments)
        assert reason is not None
        assert "end_time" in reason

    def test_segment_with_start_time_end_time_keys(self, cache_base):
        """Validates entries using start_time/end_time (legacy format)."""
        cache = TranscriptCache(str(cache_base))
        segments = [{'start_time': 0.0, 'end_time': 5.0, 'text': 'hello'}]
        assert cache.validate_entry(segments) is None


class TestGetDiscardsInvalidEntries:
    """Cache get() discards invalid entries."""

    def test_get_discards_empty_segments(self, cache_base, caplog):
        """Cache entry with empty segments list is discarded on get()."""
        cache_dir = cache_base / "transcriptions"
        _write_cache_entry(cache_dir, "abc123.json", [], source_file="test_video.mp4")

        cache = TranscriptCache(str(cache_base))

        with caplog.at_level(logging.WARNING, logger="src.transcription.cache"):
            result = cache.get(str(cache_dir / "nonexistent_but_hash_matched.mp4"))

        # Empty segments -> get() returns None after normalization produces empty list
        assert result is None

    def test_get_discards_segments_missing_text(self, cache_base, caplog):
        """Cache entry with segments missing text field is discarded on get()."""
        cache_dir = cache_base / "transcriptions"
        segments = [
            {"index": 1, "start_time": 0.0, "end_time": 5.0, "text": "", "source_file": "missing_text.mp4"}
        ]
        _write_cache_entry(cache_dir, "bad_text.json", segments, source_file="missing_text.mp4")

        cache = TranscriptCache(str(cache_base))

        with caplog.at_level(logging.WARNING, logger="src.transcription.cache"):
            result = cache.get("missing_text.mp4")

        assert result is None
        # Check that a warning was logged about the invalid entry
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("invalid cache entry" in m.lower() or "empty text" in m.lower() for m in warning_msgs), \
            f"Expected validation warning, got: {warning_msgs}"

    def test_get_returns_valid_entry(self, cache_base):
        """Valid cache entries are returned normally."""
        cache_dir = cache_base / "transcriptions"
        segments = [
            {"index": 1, "start_time": 0.0, "end_time": 5.0, "text": "hello world", "source_file": "good_video.mp4"}
        ]
        _write_cache_entry(cache_dir, "good.json", segments, source_file="good_video.mp4")

        cache = TranscriptCache(str(cache_base))
        result = cache.get("good_video.mp4")

        assert result is not None
        assert len(result) == 1
        assert result[0]['text'] == 'hello world'


class TestWarmupSkipsInvalidEntries:
    """warmup_from_project() skips invalid entries with a count summary log."""

    def test_warmup_skips_invalid_and_logs_count(self, cache_base, caplog):
        """warmup_from_project skips invalid entries and logs count."""
        # Set up global cache
        global_cache_dir = cache_base / "global"

        # Set up project cache with mix of valid and invalid entries
        project_dir = cache_base / "project"
        project_cache = project_dir / ".cache" / "transcriptions"
        project_cache.mkdir(parents=True, exist_ok=True)

        # Valid entry
        valid_data = [{"index": 1, "start_time": 0.0, "end_time": 5.0,
                       "text": "valid text", "source_file": "valid_video.mp4"}]
        (project_cache / "valid.json").write_text(json.dumps(valid_data), encoding="utf-8")

        # Invalid entry: empty segments
        (project_cache / "empty.json").write_text(json.dumps([]), encoding="utf-8")

        # Invalid entry: segment with empty text
        invalid_data = [{"index": 1, "start_time": 0.0, "end_time": 5.0,
                         "text": "", "source_file": "bad_video.mp4"}]
        (project_cache / "bad_text.json").write_text(json.dumps(invalid_data), encoding="utf-8")

        cache = TranscriptCache(str(global_cache_dir))

        with caplog.at_level(logging.WARNING, logger="src.transcription.cache"):
            imported = cache.warmup_from_project(str(project_dir))

        # Only the valid entry should be imported
        assert imported == 1

        # Should log a warning about skipped invalid entries
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("skipped" in m.lower() and "invalid" in m.lower() for m in warning_msgs), \
            f"Expected warmup invalid count warning, got: {warning_msgs}"
        # Count should be 2 (empty segments + empty text)
        assert any("2" in m for m in warning_msgs if "invalid" in m.lower()), \
            f"Expected count of 2 invalid entries, got: {warning_msgs}"

    def test_warmup_imports_all_valid(self, cache_base):
        """When all entries are valid, all are imported normally."""
        global_cache_dir = cache_base / "global"
        project_dir = cache_base / "project"
        project_cache = project_dir / ".cache" / "transcriptions"
        project_cache.mkdir(parents=True, exist_ok=True)

        for i in range(3):
            data = [{"index": 1, "start_time": 0.0, "end_time": 5.0,
                     "text": f"text {i}", "source_file": f"video_{i}.mp4"}]
            (project_cache / f"entry_{i}.json").write_text(json.dumps(data), encoding="utf-8")

        cache = TranscriptCache(str(global_cache_dir))
        imported = cache.warmup_from_project(str(project_dir))

        assert imported == 3
