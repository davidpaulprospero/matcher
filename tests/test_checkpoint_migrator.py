"""
Tests for CheckpointMigrator — extracted migration logic from CheckpointManager.

All tests use raw dict fixtures (no file I/O, no CheckpointManager dependency).
"""

import pytest
from src.checkpoint_migrator import CheckpointMigrator


@pytest.fixture
def migrator():
    return CheckpointMigrator()


class TestNeedsMigration:
    def test_needs_migration_old_version(self, migrator):
        data = {"version": "0.9"}
        assert migrator.needs_migration(data, "2.0") is True

    def test_no_migration_needed_current(self, migrator):
        data = {"version": "2.0"}
        assert migrator.needs_migration(data, "2.0") is False

    def test_missing_version_assumes_0_9(self, migrator):
        data = {}
        assert migrator.needs_migration(data, "2.0") is True


class TestMigrate0_9To1_0:
    def test_uppercase_keys_lowered(self, migrator):
        data = {
            "version": "0.9",
            "ANALYZE": {"keywords": ["test"]},
            "MATCH": {"match_count": 5},
        }
        result = migrator.migrate(data, "0.9", "1.0")
        assert result["version"] == "1.0"
        assert result["analyze"] == {"keywords": ["test"]}
        assert result["match"] == {"match_count": 5}
        assert "ANALYZE" not in result
        assert "MATCH" not in result

    def test_preserves_lowercase_keys(self, migrator):
        data = {
            "version": "0.9",
            "created_at": "2025-01-01T00:00:00",
            "last_completed_stage": "MATCH",
        }
        result = migrator.migrate(data, "0.9", "1.0")
        assert result["created_at"] == "2025-01-01T00:00:00"
        assert result["last_completed_stage"] == "MATCH"


class TestMigrate1_0To2_0:
    def test_basic_migration(self, migrator):
        data = {
            "version": "1.0",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T12:00:00",
            "config_hash": "abc123",
            "last_completed_stage": "MATCH",
            "match": {"match_count": 10},
        }
        result = migrator.migrate(data, "1.0", "2.0")
        assert result["version"] == "2.0"
        assert result["last_completed_stage"] == "MATCH"
        assert result["match"] == {"match_count": 10}

    def test_stage_remap_download(self, migrator):
        data = {"version": "1.0", "last_completed_stage": "DOWNLOAD"}
        result = migrator.migrate(data, "1.0", "2.0")
        assert result["last_completed_stage"] == "VIDEO_SEARCH"

    def test_stage_remap_transcribe(self, migrator):
        data = {"version": "1.0", "last_completed_stage": "TRANSCRIBE"}
        result = migrator.migrate(data, "1.0", "2.0")
        assert result["last_completed_stage"] == "CAPTION"

    def test_stage_remap_broll_match(self, migrator):
        data = {"version": "1.0", "last_completed_stage": "BROLL_MATCH"}
        result = migrator.migrate(data, "1.0", "2.0")
        assert result["last_completed_stage"] == "MATCH"

    def test_unknown_stage_reset(self, migrator):
        data = {"version": "1.0", "last_completed_stage": "NONEXISTENT"}
        result = migrator.migrate(data, "1.0", "2.0")
        assert result["last_completed_stage"] == ""

    def test_download_data_migrated_to_video_search(self, migrator):
        data = {
            "version": "1.0",
            "download": {
                "downloaded_videos": [
                    {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
                    {"url": "https://youtu.be/abcdefghijk"},
                ]
            },
        }
        result = migrator.migrate(data, "1.0", "2.0")
        vs = result["video_search"]
        assert "dQw4w9WgXcQ" in vs["video_ids"]
        assert "abcdefghijk" in vs["video_ids"]
        assert vs["migrated_from_download"] is True

    def test_preserves_chapter_data(self, migrator):
        data = {
            "version": "1.0",
            "chapter_data": {"chapters": [{"title": "Intro"}]},
        }
        result = migrator.migrate(data, "1.0", "2.0")
        assert result["chapter_data"] == {"chapters": [{"title": "Intro"}]}


class TestChainedMigration:
    def test_0_9_to_2_0_chains(self, migrator):
        """v0.9 -> v2.0 should chain through v1.0 automatically."""
        data = {
            "version": "0.9",
            "ANALYZE": {"keywords": ["test"]},
            "last_completed_stage": "DOWNLOAD",
            "DOWNLOAD": {
                "downloaded_videos": [
                    {"url": "https://youtube.com/watch?v=xxxxxxxxxxx"}
                ]
            },
        }
        result = migrator.migrate(data, "0.9", "2.0")
        assert result["version"] == "2.0"
        assert result["last_completed_stage"] == "VIDEO_SEARCH"
        assert result["analyze"] == {"keywords": ["test"]}
        assert "xxxxxxxxxxx" in result["video_search"]["video_ids"]

    def test_no_op_same_version(self, migrator):
        data = {"version": "2.0", "last_completed_stage": "MATCH"}
        result = migrator.migrate(data, "2.0", "2.0")
        assert result == data


class TestEdgeCases:
    def test_unknown_version_raises(self, migrator):
        with pytest.raises(ValueError, match="Unknown version"):
            migrator.migrate({}, "0.5", "2.0")

    def test_backwards_migration_raises(self, migrator):
        with pytest.raises(ValueError, match="Cannot migrate backwards"):
            migrator.migrate({}, "2.0", "0.9")

    def test_empty_data_migrates(self, migrator):
        """Empty dict should migrate without errors."""
        result = migrator.migrate({}, "0.9", "2.0")
        assert result["version"] == "2.0"

    def test_does_not_mutate_input(self, migrator):
        data = {"version": "1.0", "last_completed_stage": "DOWNLOAD"}
        original = dict(data)
        migrator.migrate(data, "1.0", "2.0")
        assert data == original


class TestFutureMigrationExtensibility:
    """Verify that adding a new migration is straightforward."""

    def test_adding_migration_is_one_function(self, migrator):
        """Docstring documents that adding v2.0->v3.0 requires one function."""
        # Verify the class has the expected structure
        assert hasattr(migrator, '_migrations')
        assert hasattr(migrator, '_version_chain')
        # The migration registry is a dict, so adding one entry is sufficient
        assert isinstance(migrator._migrations, dict)
