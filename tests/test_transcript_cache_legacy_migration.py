"""
Tests for legacy transcripts/ cache directory deprecation warning and migration.

US-66-010: TranscriptCache warns when legacy transcripts/ directory has files
and automatically migrates entries to transcriptions/.
"""

import json
import logging
import sys
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transcription.cache import TranscriptCache

pytestmark = [pytest.mark.fast, pytest.mark.unit]


@pytest.fixture
def cache_base(tmp_path):
    """Return a tmp_path to use as the base cache directory."""
    return tmp_path


def _write_cache_entry(directory: Path, filename: str, source_file: str = "video.mp4"):
    """Helper: write a minimal cache JSON file."""
    directory.mkdir(parents=True, exist_ok=True)
    data = [{"index": 1, "start_time": 0.0, "end_time": 5.0, "text": "hello", "source_file": source_file}]
    path = directory / filename
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestNoWarningWhenOnlyPrimaryExists:
    """No warning when only transcriptions/ exists (no legacy dir)."""

    def test_no_warning_only_transcriptions(self, cache_base, caplog):
        """No warning logged when transcripts/ does not exist."""
        _write_cache_entry(cache_base / "transcriptions", "abc123.json")

        with caplog.at_level(logging.WARNING, logger="src.transcription.cache"):
            cache = TranscriptCache(str(cache_base))

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("transcripts/" in m for m in warning_msgs)

    def test_no_warning_empty_legacy_dir(self, cache_base, caplog):
        """No warning when transcripts/ exists but is empty."""
        (cache_base / "transcriptions").mkdir(parents=True, exist_ok=True)
        (cache_base / "transcripts").mkdir(parents=True, exist_ok=True)

        with caplog.at_level(logging.WARNING, logger="src.transcription.cache"):
            cache = TranscriptCache(str(cache_base))

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("transcripts/" in m for m in warning_msgs)


class TestWarningWhenLegacyHasFiles:
    """Warning logged when transcripts/ contains cached files."""

    def test_warning_logged_with_legacy_files(self, cache_base, caplog):
        """Warning includes migration guidance when transcripts/ has files."""
        _write_cache_entry(cache_base / "transcripts", "legacy1.json", "old_video.mp4")
        # Ensure primary dir exists (created by __init__)

        with caplog.at_level(logging.WARNING, logger="src.transcription.cache"):
            cache = TranscriptCache(str(cache_base))

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        legacy_warnings = [m for m in warning_msgs if "transcripts/" in m]
        assert len(legacy_warnings) >= 1, f"Expected legacy warning, got: {warning_msgs}"

        msg = legacy_warnings[0]
        assert "Move files from transcripts/ to transcriptions/" in msg
        assert "delete transcripts/" in msg

    def test_warning_includes_file_count(self, cache_base, caplog):
        """Warning message mentions the number of legacy files."""
        legacy_dir = cache_base / "transcripts"
        _write_cache_entry(legacy_dir, "a.json", "vid_a.mp4")
        _write_cache_entry(legacy_dir, "b.json", "vid_b.mp4")

        with caplog.at_level(logging.WARNING, logger="src.transcription.cache"):
            cache = TranscriptCache(str(cache_base))

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        legacy_warnings = [m for m in warning_msgs if "transcripts/" in m]
        assert any("2" in m for m in legacy_warnings), f"Expected '2' in warning, got: {legacy_warnings}"


class TestMigrationCopiesFiles:
    """Migration copies entries from transcripts/ to transcriptions/."""

    def test_migration_copies_missing_files(self, cache_base):
        """Files in transcripts/ are copied to transcriptions/ when not already present."""
        _write_cache_entry(cache_base / "transcripts", "legacy.json", "legacy_vid.mp4")

        cache = TranscriptCache(str(cache_base))

        # The file should now exist in transcriptions/
        dest = cache_base / "transcriptions" / "legacy.json"
        assert dest.exists(), "Legacy file should be copied to transcriptions/"

        # Verify content matches
        with open(dest, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data[0]["source_file"] == "legacy_vid.mp4"

    def test_migration_skips_existing_files(self, cache_base):
        """Files already in transcriptions/ are not overwritten."""
        # Write different content in primary vs legacy with same filename
        primary_dir = cache_base / "transcriptions"
        primary_dir.mkdir(parents=True, exist_ok=True)
        _write_cache_entry(primary_dir, "same.json", "primary_video.mp4")
        _write_cache_entry(cache_base / "transcripts", "same.json", "legacy_video.mp4")

        cache = TranscriptCache(str(cache_base))

        # Primary should keep its original content
        with open(primary_dir / "same.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data[0]["source_file"] == "primary_video.mp4"

    def test_migration_runs_once_per_session(self, cache_base):
        """Calling _migrate_legacy_cache() again is a no-op after first run."""
        _write_cache_entry(cache_base / "transcripts", "leg.json", "v.mp4")

        cache = TranscriptCache(str(cache_base))
        assert cache._legacy_migrated is True

        # Remove the copied file to prove second call is a no-op
        copied = cache_base / "transcriptions" / "leg.json"
        assert copied.exists()
        copied.unlink()

        cache._migrate_legacy_cache()

        # Should NOT re-copy because guard flag prevents it
        assert not copied.exists(), "Second migration call should be a no-op"

    def test_migration_multiple_files(self, cache_base):
        """Multiple legacy files are all migrated."""
        legacy_dir = cache_base / "transcripts"
        for i in range(5):
            _write_cache_entry(legacy_dir, f"file_{i}.json", f"video_{i}.mp4")

        cache = TranscriptCache(str(cache_base))

        primary_dir = cache_base / "transcriptions"
        for i in range(5):
            assert (primary_dir / f"file_{i}.json").exists(), f"file_{i}.json should be migrated"
