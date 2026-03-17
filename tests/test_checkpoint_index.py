"""
Test checkpoint metadata indexing (US-130-009).

Tests checkpoint.index file creation, metadata tracking,
query methods, and atomic updates.
"""
import json
import pytest
import shutil
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta

from src.checkpoint_index import CheckpointIndex, IndexEntry, get_or_create_index


class TestCheckpointIndex:
    """Tests for checkpoint metadata indexing."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def index(self, temp_project_dir):
        """Create a CheckpointIndex for testing."""
        return CheckpointIndex(project_dir=temp_project_dir)

    def test_index_path(self, temp_project_dir):
        """Test that index path is correctly set."""
        index = CheckpointIndex(project_dir=temp_project_dir)
        assert index.index_path == temp_project_dir / "checkpoint.index"

    def test_empty_index_load(self, index):
        """Test loading empty/non-existent index."""
        index.load()
        assert index.entries == []

    def test_add_entry(self, index):
        """Test adding index entry."""
        index.add_entry(
            stage="MATCH",
            size=1024,
            config_hash="abc123",
            voiceover_hash="def456"
        )
        assert len(index.entries) == 1
        entry = index.entries[0]
        assert entry.stage == "MATCH"
        assert entry.size == 1024
        assert entry.config_hash == "abc123"
        assert entry.voiceover_hash == "def456"
        assert entry.timestamp is not None

    def test_atomic_save_and_load(self, index):
        """Test atomic save and load of index."""
        # Add entry
        index.add_entry(
            stage="DOWNLOAD_SEGMENTS",
            size=2048,
            config_hash="xyz789",
            voiceover_hash="uvw012"
        )

        # Save
        index._acquire_lock()
        try:
            index._atomic_save()
        finally:
            index._release_lock()

        # Load in new index instance
        new_index = CheckpointIndex(project_dir=index.project_dir)
        new_index.load()

        assert len(new_index.entries) == 1
        entry = new_index.entries[0]
        assert entry.stage == "DOWNLOAD_SEGMENTS"
        assert entry.size == 2048

    def test_query_by_stage(self, index):
        """Test querying entries by stage name."""
        # Add multiple entries
        index.entries = [
            IndexEntry(timestamp=datetime.now().isoformat(), stage="ANALYZE", size=100,
                      config_hash="a", voiceover_hash="b"),
            IndexEntry(timestamp=datetime.now().isoformat(), stage="MATCH", size=200,
                      config_hash="a", voiceover_hash="b"),
            IndexEntry(timestamp=datetime.now().isoformat(), stage="MATCH", size=300,
                      config_hash="a", voiceover_hash="b"),
            IndexEntry(timestamp=datetime.now().isoformat(), stage="DOWNLOAD_SEGMENTS", size=400,
                      config_hash="a", voiceover_hash="b"),
        ]

        results = index.query_by_stage("MATCH")
        assert len(results) == 2

        results = index.query_by_stage("ANALYZE")
        assert len(results) == 1

        results = index.query_by_stage("OUTPUT")
        assert len(results) == 0

    def test_query_by_date_range(self, index):
        """Test querying entries by date range."""
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)

        # Use fixed timestamps for testing
        index.entries = [
            IndexEntry(timestamp=yesterday.isoformat(), stage="ANALYZE", size=100,
                      config_hash="a", voiceover_hash="b"),
            IndexEntry(timestamp=now.isoformat(), stage="MATCH", size=200,
                      config_hash="a", voiceover_hash="b"),
            IndexEntry(timestamp=tomorrow.isoformat(), stage="DOWNLOAD_SEGMENTS", size=300,
                      config_hash="a", voiceover_hash="b"),
        ]

        # Query for entries in the last 2 days (includes yesterday and today)
        results = index.query_by_date_range(start=yesterday, end=now)
        assert len(results) == 2

        # Query for entries up to and including tomorrow
        results = index.query_by_date_range(start=yesterday, end=tomorrow)
        assert len(results) == 3

        # Query for only yesterday
        # Add microseconds to make range exclusive work correctly
        yesterday_end = yesterday + timedelta(microseconds=1)
        results = index.query_by_date_range(start=yesterday, end=yesterday_end)
        assert len(results) == 1

    def test_query_by_config_hash(self, index):
        """Test querying entries by config hash."""
        index.entries = [
            IndexEntry(timestamp=datetime.now().isoformat(), stage="ANALYZE", size=100,
                      config_hash="abc123", voiceover_hash="b"),
            IndexEntry(timestamp=datetime.now().isoformat(), stage="MATCH", size=200,
                      config_hash="abc123", voiceover_hash="b"),
            IndexEntry(timestamp=datetime.now().isoformat(), stage="DOWNLOAD_SEGMENTS", size=300,
                      config_hash="xyz789", voiceover_hash="b"),
        ]

        results = index.query_by_config_hash("abc123")
        assert len(results) == 2

        results = index.query_by_config_hash("xyz789")
        assert len(results) == 1

        results = index.query_by_config_hash("nonexistent")
        assert len(results) == 0

    def test_get_latest(self, index):
        """Test getting the latest index entry."""
        index.entries = [
            IndexEntry(timestamp="2024-01-01T00:00:00", stage="ANALYZE", size=100,
                      config_hash="a", voiceover_hash="b"),
            IndexEntry(timestamp="2024-01-02T00:00:00", stage="MATCH", size=200,
                      config_hash="a", voiceover_hash="b"),
            IndexEntry(timestamp="2024-01-03T00:00:00", stage="DOWNLOAD_SEGMENTS", size=300,
                      config_hash="a", voiceover_hash="b"),
        ]

        latest = index.get_latest()
        assert latest.stage == "DOWNLOAD_SEGMENTS"

    def test_get_all(self, index):
        """Test getting all entries."""
        index.entries = [
            IndexEntry(timestamp=datetime.now().isoformat(), stage="ANALYZE", size=100,
                      config_hash="a", voiceover_hash="b"),
            IndexEntry(timestamp=datetime.now().isoformat(), stage="MATCH", size=200,
                      config_hash="a", voiceover_hash="b"),
        ]

        all_entries = index.get_all()
        assert len(all_entries) == 2

        # Verify it's a copy
        all_entries.clear()
        assert len(index.entries) == 2

    def test_clear(self, index):
        """Test clearing index entries."""
        index.entries = [
            IndexEntry(timestamp=datetime.now().isoformat(), stage="ANALYZE", size=100,
                      config_hash="a", voiceover_hash="b"),
        ]

        index.clear()
        assert len(index.entries) == 0

    def test_index_entry_to_dict(self):
        """Test IndexEntry serialization."""
        entry = IndexEntry(
            timestamp="2024-01-01T12:00:00",
            stage="MATCH",
            size=1024,
            config_hash="abc123",
            voiceover_hash="def456",
            checkpoint_file="checkpoint.json"
        )

        data = entry.to_dict()
        assert data["timestamp"] == "2024-01-01T12:00:00"
        assert data["stage"] == "MATCH"
        assert data["size"] == 1024
        assert data["config_hash"] == "abc123"
        assert data["voiceover_hash"] == "def456"

    def test_index_entry_from_dict(self):
        """Test IndexEntry deserialization."""
        data = {
            "timestamp": "2024-01-01T12:00:00",
            "stage": "MATCH",
            "size": 1024,
            "config_hash": "abc123",
            "voiceover_hash": "def456",
            "checkpoint_file": "checkpoint.json"
        }

        entry = IndexEntry.from_dict(data)
        assert entry.timestamp == "2024-01-01T12:00:00"
        assert entry.stage == "MATCH"
        assert entry.size == 1024
        assert entry.config_hash == "abc123"
        assert entry.voiceover_hash == "def456"

    def test_get_or_create_index(self, temp_project_dir):
        """Test get_or_create_index helper function."""
        index = get_or_create_index(temp_project_dir)
        assert isinstance(index, CheckpointIndex)
        assert index.project_dir == temp_project_dir
        assert index.entries == []

    def test_add_entry_and_save(self, index):
        """Test add_entry_and_save method."""
        index.add_entry_and_save(
            stage="MATCH",
            size=1024,
            config_hash="test123",
            voiceover_hash="test456"
        )

        # Load in new instance
        new_index = CheckpointIndex(project_dir=index.project_dir)
        new_index.load()

        assert len(new_index.entries) == 1
        assert new_index.entries[0].stage == "MATCH"

    def test_multiple_entries_ordering(self, index):
        """Test that multiple entries are kept in order."""
        stages = ["ANALYZE", "VIDEO_SEARCH", "CAPTION", "MATCH", "ITERATIVE_MATCH", "DOWNLOAD_SEGMENTS"]
        for i, stage in enumerate(stages):
            index.add_entry_and_save(
                stage=stage,
                size=100 * (i + 1),
                config_hash=f"hash{i}",
                voiceover_hash="vo_hash"
            )

        new_index = CheckpointIndex(project_dir=index.project_dir)
        new_index.load()

        assert len(new_index.entries) == 6
        # Verify ordering
        for i, entry in enumerate(new_index.entries):
            assert entry.stage == stages[i]
