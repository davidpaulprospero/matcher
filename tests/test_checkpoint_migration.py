"""
Tests for Checkpoint Migration — US-108-004

Verifies migration from v2.0 to v2.1 preserves all critical checkpoint data.
"""

import pytest
from src.checkpoint_migrator import CheckpointMigrator
from src.checkpoint import CheckpointData, CURRENT_CHECKPOINT_VERSION


@pytest.fixture
def migrator():
    return CheckpointMigrator()


class TestMigrate2_0To2_1:
    """Test migration from version 2.0 to 2.1."""

    def test_basic_migration(self, migrator):
        """Basic migration preserves all critical fields."""
        data = {
            "version": "2.0",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T12:00:00",
            "config_hash": "abc123",
            "voiceover_path": "/path/to/voiceover.mp3",
            "voiceover_hash": "def456",
            "last_completed_stage": "MATCH",
            "analyze": {"keywords": ["test"], "segment_count": 5},
            "video_search": {"video_ids": ["vid1", "vid2"]},
            "caption": {"caption_count": 2},
            "match": {"match_count": 10, "avg_confidence": 0.85},
            "iterative_match": {"additional_matches": 3},
            "download_segments": {"downloaded_segments": 8},
            "chapter_data": {"chapters": [{"title": "Intro"}]},
            "stage_metrics": {"MATCH": {"items_processed": 10}},
            "transcription_metrics": {"total_videos": 5},
        }
        result = migrator.migrate(data, "2.0", "2.1")

        # Version should be updated
        assert result["version"] == "2.1"

        # Critical fields should be preserved
        assert result["created_at"] == "2025-01-01T00:00:00"
        assert result["updated_at"] == "2025-01-01T12:00:00"
        assert result["config_hash"] == "abc123"
        assert result["voiceover_path"] == "/path/to/voiceover.mp3"
        assert result["voiceover_hash"] == "def456"
        assert result["last_completed_stage"] == "MATCH"

        # Stage data should be preserved
        assert result["analyze"] == {"keywords": ["test"], "segment_count": 5}
        assert result["video_search"] == {"video_ids": ["vid1", "vid2"]}
        assert result["caption"] == {"caption_count": 2}
        assert result["match"] == {"match_count": 10, "avg_confidence": 0.85}
        assert result["iterative_match"] == {"additional_matches": 3}
        assert result["download_segments"] == {"downloaded_segments": 8}
        assert result["chapter_data"] == {"chapters": [{"title": "Intro"}]}
        assert result["stage_metrics"] == {"MATCH": {"items_processed": 10}}
        assert result["transcription_metrics"] == {"total_videos": 5}

    def test_adds_new_fields_with_defaults(self, migrator):
        """Migration adds new v2.1 fields with empty defaults."""
        data = {"version": "2.0", "last_completed_stage": "CAPTION"}
        result = migrator.migrate(data, "2.0", "2.1")

        # New fields should be added with defaults
        assert "validation_cache" in result
        assert result["validation_cache"] == {}
        assert "escalation_state" in result
        assert result["escalation_state"] == {}
        assert "circuit_breaker_health" in result
        assert result["circuit_breaker_health"] == {}

    def test_preserves_existing_new_fields(self, migrator):
        """Migration preserves new fields if they already exist."""
        data = {
            "version": "2.0",
            "validation_cache": {"MATCH": {"is_valid": True}},
            "escalation_state": {"keyword_states": {"test": {"tier": 2}}},
            "circuit_breaker_health": {"trip_count": 5},
        }
        result = migrator.migrate(data, "2.0", "2.1")

        # Existing values should be preserved
        assert result["validation_cache"] == {"MATCH": {"is_valid": True}}
        assert result["escalation_state"] == {"keyword_states": {"test": {"tier": 2}}}
        assert result["circuit_breaker_health"] == {"trip_count": 5}

    def test_adds_migration_metadata(self, migrator):
        """Migration adds metadata about the migration."""
        data = {"version": "2.0", "last_completed_stage": "VIDEO_SEARCH"}
        result = migrator.migrate(data, "2.0", "2.1")

        assert "_last_migration" in result
        assert result["_last_migration"]["from_version"] == "2.0"
        assert result["_last_migration"]["to_version"] == "2.1"
        assert "migrated_at" in result["_last_migration"]


class TestChainedMigrationTo2_1:
    """Test that older versions can chain to v2.1."""

    def test_0_9_to_2_1_chains(self, migrator):
        """v0.9 -> v2.1 should chain through all intermediate versions."""
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
        result = migrator.migrate(data, "0.9", "2.1")

        assert result["version"] == "2.1"
        assert result["last_completed_stage"] == "VIDEO_SEARCH"
        assert result["analyze"] == {"keywords": ["test"]}
        assert "xxxxxxxxxxx" in result["video_search"]["video_ids"]

    def test_1_0_to_2_1_chains(self, migrator):
        """v1.0 -> v2.1 should chain through v2.0."""
        data = {
            "version": "1.0",
            "last_completed_stage": "MATCH",
            "match": {"match_count": 5},
        }
        result = migrator.migrate(data, "1.0", "2.1")

        assert result["version"] == "2.1"
        assert result["last_completed_stage"] == "MATCH"
        assert result["match"] == {"match_count": 5}

    def test_2_0_to_2_1_direct(self, migrator):
        """v2.0 -> v2.1 should migrate directly."""
        data = {"version": "2.0", "last_completed_stage": "CAPTION"}
        result = migrator.migrate(data, "2.0", "2.1")

        assert result["version"] == "2.1"
        assert result["last_completed_stage"] == "CAPTION"


class TestMigrationPreservesCriticalData:
    """Verify migration preserves all critical checkpoint data."""

    def test_preserves_all_stage_data(self, migrator):
        """All stage data should be preserved through migration."""
        data = {
            "version": "2.0",
            "last_completed_stage": "DOWNLOAD_SEGMENTS",
            "analyze": {
                "keywords": ["AI", "technology", "future"],
                "segments": [
                    {"start": 0, "end": 30, "text": "Introduction"},
                    {"start": 30, "end": 60, "text": "Main content"},
                ],
                "segment_count": 2,
            },
            "video_search": {
                "video_ids": ["abc123", "def456", "ghi789"],
                "search_queries": ["AI technology"],
            },
            "caption": {
                "captions": {"abc123": "sample caption", "def456": "another"},
                "caption_count": 2,
            },
            "match": {
                "matches": [
                    {"segment_idx": 0, "video_id": "abc123", "confidence": 0.9},
                    {"segment_idx": 1, "video_id": "def456", "confidence": 0.85},
                ],
                "match_count": 2,
                "avg_confidence": 0.875,
            },
            "iterative_match": {
                "gap_analysis": [{"segment_idx": 2, "gap_reason": "no_match"}],
                "additional_matches": 1,
            },
            "download_segments": {
                "downloaded_segments": [
                    {"video_id": "abc123", "start": 0, "end": 30, "path": "/path/1.mp4"}
                ],
                "download_results": {"success": 1, "failed": 0},
            },
        }
        result = migrator.migrate(data, "2.0", "2.1")

        # All stage data should be identical
        assert result["analyze"] == data["analyze"]
        assert result["video_search"] == data["video_search"]
        assert result["caption"] == data["caption"]
        assert result["match"] == data["match"]
        assert result["iterative_match"] == data["iterative_match"]
        assert result["download_segments"] == data["download_segments"]

    def test_preserves_chapter_data(self, migrator):
        """Chapter data should be preserved."""
        data = {
            "version": "2.0",
            "chapter_data": {
                "chapters": [
                    {"title": "Introduction", "start_time": 0, "end_time": 60},
                    {"title": "Main Topic", "start_time": 60, "end_time": 180},
                ],
                "listicle_groups": [
                    {"topic": "Top 5 AI Trends", "items": ["a", "b", "c"]}
                ],
            },
        }
        result = migrator.migrate(data, "2.0", "2.1")

        assert result["chapter_data"] == data["chapter_data"]

    def test_preserves_stage_metrics(self, migrator):
        """Stage metrics should be preserved."""
        data = {
            "version": "2.0",
            "stage_metrics": {
                "ANALYZE": {"duration_seconds": 5.2, "items_processed": 3},
                "VIDEO_SEARCH": {"duration_seconds": 12.3, "items_processed": 50},
                "MATCH": {"duration_seconds": 30.5, "items_processed": 10},
                "_pipeline": {"total_duration_seconds": 120.0},
            },
        }
        result = migrator.migrate(data, "2.0", "2.1")

        assert result["stage_metrics"] == data["stage_metrics"]

    def test_preserves_transcription_metrics(self, migrator):
        """Transcription metrics should be preserved."""
        data = {
            "version": "2.0",
            "transcription_metrics": {
                "total_videos": 100,
                "cached_videos": 45,
                "transcribed_videos": 50,
                "failed_videos": 5,
                "total_duration_seconds": 3600.0,
                "average_speed_ratio": 15.5,
            },
        }
        result = migrator.migrate(data, "2.0", "2.1")

        assert result["transcription_metrics"] == data["transcription_metrics"]


class TestCheckpointMigratorIntegration:
    """Integration tests for CheckpointMigrator with CheckpointData."""

    def test_migrated_data_loads_as_checkpoint_data(self, migrator):
        """Migrated data should be loadable as CheckpointData."""
        data = {
            "version": "2.0",
            "created_at": "2025-01-01T00:00:00",
            "updated_at": "2025-01-01T12:00:00",
            "config_hash": "abc123",
            "last_completed_stage": "MATCH",
            "analyze": {"keywords": ["test"]},
        }
        migrated = migrator.migrate(data, "2.0", "2.1")

        # Should be able to create CheckpointData from migrated dict
        cp_data = CheckpointData.from_dict(migrated)

        assert cp_data.version == "2.1"
        assert cp_data.last_completed_stage == "MATCH"
        assert cp_data.analyze == {"keywords": ["test"]}

    def test_current_version_is_2_1(self):
        """Verify CURRENT_CHECKPOINT_VERSION is 2.1."""
        assert CURRENT_CHECKPOINT_VERSION == "2.1"

    def test_needs_migration_2_0_to_2_1(self, migrator):
        """Check that migration is needed from 2.0 to 2.1."""
        data = {"version": "2.0"}
        assert migrator.needs_migration(data, "2.1") is True

    def test_no_migration_needed_for_2_1(self, migrator):
        """Check that no migration is needed for 2.1 to 2.1."""
        data = {"version": "2.1"}
        assert migrator.needs_migration(data, "2.1") is False


class TestMigrationEdgeCases:
    """Edge case tests for v2.0 to v2.1 migration."""

    def test_empty_checkpoint_migrates(self, migrator):
        """Empty checkpoint should migrate without errors."""
        result = migrator.migrate({}, "2.0", "2.1")
        assert result["version"] == "2.1"
        assert result["validation_cache"] == {}
        assert result["escalation_state"] == {}
        assert result["circuit_breaker_health"] == {}

    def test_partial_checkpoint_migrates(self, migrator):
        """Partial checkpoint with only required fields should migrate."""
        data = {"version": "2.0"}
        result = migrator.migrate(data, "2.0", "2.1")

        assert result["version"] == "2.1"
        # New fields should have defaults
        assert "validation_cache" in result
        assert "escalation_state" in result
        assert "circuit_breaker_health" in result

    def test_does_not_mutate_input(self, migrator):
        """Migration should not mutate the input dict."""
        data = {"version": "2.0", "last_completed_stage": "MATCH"}
        original = dict(data)
        migrator.migrate(data, "2.0", "2.1")
        assert data == original
