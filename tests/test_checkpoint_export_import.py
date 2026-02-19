"""
Test checkpoint export/import functionality (US-115-009).

Tests:
- Export checkpoint to JSON
- Import checkpoint from JSON
- Selective stage export
- Round-trip data integrity
- Compressed export
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from src.checkpoint import CheckpointManager, CheckpointData, STAGE_ORDER, CURRENT_CHECKPOINT_VERSION


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_checkpoint_data():
    """Create sample checkpoint data for testing."""
    data = CheckpointData(
        version=CURRENT_CHECKPOINT_VERSION,
        last_completed_stage="MATCH",
        voiceover_path="/path/to/voiceover.srt",
        voiceover_hash="abc123",
        analyze={"keywords": ["test", "video"]},
        video_search={"videos": [{"id": "vid1", "title": "Test Video"}]},
        caption={"captions": {"vid1": {"text": "Hello world"}}},
        match={"segments": [{"id": "seg1", "video_id": "vid1"}]},
        iterative_match={},
        download_segments={},
        stage_metrics={
            "VIDEO_SEARCH": {"duration": 1.5, "videos_found": 10},
            "CAPTION": {"duration": 2.0, "captions_fetched": 5},
        },
    )
    return data


class TestCheckpointExportImport:
    """Test checkpoint export/import functionality."""

    def test_export_checkpoint_full(self, temp_project_dir, sample_checkpoint_data):
        """Test full checkpoint export."""
        # Create checkpoint manager and save sample data
        checkpoint_mgr = CheckpointManager(temp_project_dir, config_hash="test_hash")
        checkpoint_mgr.data = sample_checkpoint_data
        checkpoint_mgr.save("MATCH", sample_checkpoint_data.to_dict())

        # Export
        export_path = temp_project_dir / "exported.json"
        success = checkpoint_mgr.export_checkpoint(str(export_path))

        assert success
        assert export_path.exists()

        # Verify export contents
        with open(export_path) as f:
            exported = json.load(f)

        assert exported['version'] == CURRENT_CHECKPOINT_VERSION
        assert exported['last_completed_stage'] == "MATCH"
        assert 'video_search' in exported
        assert 'caption' in exported
        assert 'match' in exported
        assert '_export_info' in exported
        assert exported['_export_info']['included_stages'] == 'all'

    def test_export_checkpoint_selective_stages(self, temp_project_dir, sample_checkpoint_data):
        """Test selective stage export."""
        # Create checkpoint manager and save sample data
        checkpoint_mgr = CheckpointManager(temp_project_dir, config_hash="test_hash")
        checkpoint_mgr.data = sample_checkpoint_data
        checkpoint_mgr.save("MATCH", sample_checkpoint_data.to_dict())

        # Export only video_search and caption stages
        export_path = temp_project_dir / "exported_selective.json"
        success = checkpoint_mgr.export_checkpoint(
            str(export_path),
            include_stages=['VIDEO_SEARCH', 'CAPTION']
        )

        assert success
        assert export_path.exists()

        # Verify export contents
        with open(export_path) as f:
            exported = json.load(f)

        # Should have export info
        assert '_export_info' in exported
        assert 'VIDEO_SEARCH' in exported['_export_info']['included_stages']
        assert 'CAPTION' in exported['_export_info']['included_stages']

        # Selected stages should have data
        assert exported.get('video_search') == sample_checkpoint_data.video_search
        assert exported.get('caption') == sample_checkpoint_data.caption

        # Non-selected stages should be empty dicts
        assert exported.get('match', {}) == {}

    def test_import_checkpoint(self, temp_project_dir, sample_checkpoint_data):
        """Test checkpoint import."""
        # First, export to a file
        export_path = temp_project_dir / "exported.json"
        checkpoint_mgr = CheckpointManager(temp_project_dir, config_hash="test_hash")
        checkpoint_mgr.data = sample_checkpoint_data
        checkpoint_mgr.save("MATCH", sample_checkpoint_data.to_dict())
        checkpoint_mgr.export_checkpoint(str(export_path))

        # Now create a new project directory and import
        new_project_dir = temp_project_dir / "new_project"
        new_project_dir.mkdir()

        new_checkpoint_mgr = CheckpointManager(new_project_dir, config_hash="test_hash")
        success = new_checkpoint_mgr.import_checkpoint(str(export_path), str(new_project_dir))

        assert success

        # Verify imported data
        imported_data = new_checkpoint_mgr.load()
        assert imported_data is not None
        assert imported_data.last_completed_stage == "MATCH"
        assert imported_data.video_search == sample_checkpoint_data.video_search

    def test_roundtrip_data_integrity(self, temp_project_dir, sample_checkpoint_data):
        """Test that data survives export/import round-trip."""
        # Create and save checkpoint
        checkpoint_mgr = CheckpointManager(temp_project_dir, config_hash="test_hash")
        checkpoint_mgr.data = sample_checkpoint_data
        checkpoint_mgr.save("MATCH", sample_checkpoint_data.to_dict())

        # Export
        export_path = temp_project_dir / "roundtrip.json"
        checkpoint_mgr.export_checkpoint(str(export_path))

        # Import to new directory
        new_project_dir = temp_project_dir / "roundtrip_project"
        new_project_dir.mkdir()
        new_checkpoint_mgr = CheckpointManager(new_project_dir, config_hash="test_hash")
        new_checkpoint_mgr.import_checkpoint(str(export_path), str(new_project_dir))

        # Compare data
        original = sample_checkpoint_data.to_dict()
        imported = new_checkpoint_mgr.load().to_dict()

        # Remove export-specific metadata for comparison
        original.pop('_export_info', None)
        imported.pop('_export_info', None)

        # Verify key fields match
        assert imported['version'] == original['version']
        assert imported['last_completed_stage'] == original['last_completed_stage']
        assert imported['video_search'] == original['video_search']
        assert imported['caption'] == original['caption']
        assert imported['match'] == original['match']

    def test_export_compressed(self, temp_project_dir, sample_checkpoint_data):
        """Test compressed checkpoint export."""
        import gzip

        checkpoint_mgr = CheckpointManager(temp_project_dir, config_hash="test_hash")
        checkpoint_mgr.data = sample_checkpoint_data
        checkpoint_mgr.save("MATCH", sample_checkpoint_data.to_dict())

        export_path = temp_project_dir / "exported.gz"
        success = checkpoint_mgr.export_checkpoint_compressed(str(export_path))

        assert success
        assert export_path.exists()

        # Verify it's actually gzip compressed
        with gzip.open(export_path, 'rt') as f:
            imported = json.load(f)

        assert imported['version'] == CURRENT_CHECKPOINT_VERSION
        assert imported['_export_info']['compressed'] is True

    def test_export_empty_checkpoint(self, temp_project_dir):
        """Test exporting when no checkpoint exists."""
        checkpoint_mgr = CheckpointManager(temp_project_dir, config_hash="test_hash")

        export_path = temp_project_dir / "exported.json"
        success = checkpoint_mgr.export_checkpoint(str(export_path))

        assert not success

    def test_import_nonexistent_file(self, temp_project_dir):
        """Test importing from nonexistent file."""
        checkpoint_mgr = CheckpointManager(temp_project_dir, config_hash="test_hash")

        success = checkpoint_mgr.import_checkpoint("/nonexistent/file.json")

        assert not success

    def test_cli_export_stages_argument(self, temp_project_dir, sample_checkpoint_data):
        """Test CLI --export-stages parsing."""
        checkpoint_mgr = CheckpointManager(temp_project_dir, config_hash="test_hash")
        checkpoint_mgr.data = sample_checkpoint_data
        checkpoint_mgr.save("MATCH", sample_checkpoint_data.to_dict())

        # Test with comma-separated stages
        export_path = temp_project_dir / "exported.json"
        success = checkpoint_mgr.export_checkpoint(
            str(export_path),
            include_stages=['video_search', 'caption']  # lowercase should work
        )

        assert success

        with open(export_path) as f:
            exported = json.load(f)

        # These stages should have data
        assert exported.get('video_search', {}) != {}
        assert exported.get('caption', {}) != {}

    def test_stage_filtering_case_insensitive(self, temp_project_dir, sample_checkpoint_data):
        """Test that stage filtering is case insensitive."""
        checkpoint_mgr = CheckpointManager(temp_project_dir, config_hash="test_hash")
        checkpoint_mgr.data = sample_checkpoint_data
        checkpoint_mgr.save("MATCH", sample_checkpoint_data.to_dict())

        export_path = temp_project_dir / "exported.json"

        # Test lowercase
        checkpoint_mgr.export_checkpoint(str(export_path), include_stages=['video_search'])

        with open(export_path) as f:
            exported = json.load(f)

        assert exported.get('video_search', {}) != {}


class TestCheckpointRedaction:
    """Test checkpoint redaction functionality (US-130-011)."""

    def test_redact_youtube_urls(self, temp_project_dir):
        """Test that YouTube URLs are properly redacted."""
        # Create checkpoint with URLs
        data = CheckpointData(
            version=CURRENT_CHECKPOINT_VERSION,
            last_completed_stage="MATCH",
            voiceover_path="/path/to/voiceover.srt",
            voiceover_hash="abc123",
            analyze={"keywords": ["test"]},
            video_search={
                "videos": [
                    {"id": "dQw4w9WgXcQ", "url": "https://youtube.com/watch?v=dQw4w9WgXcQ"},
                    {"id": "abc123", "url": "https://youtu.be/abc123"},
                ]
            },
            caption={},
            match={},
            iterative_match={},
            download_segments={},
            stage_metrics={},
        )

        checkpoint_mgr = CheckpointManager(temp_project_dir, config_hash="test_hash")
        checkpoint_mgr.data = data
        checkpoint_mgr.save("MATCH", data.to_dict())

        # Export with redaction
        export_path = temp_project_dir / "redacted.json"
        success = checkpoint_mgr.export_checkpoint_redacted(str(export_path))

        assert success

        with open(export_path) as f:
            exported = json.load(f)

        # Verify URLs are redacted
        videos = exported.get('video_search', {}).get('videos', [])
        for video in videos:
            assert 'url' not in video or 'REDACTED' in str(video.get('url', ''))

    def test_redact_file_paths(self, temp_project_dir):
        """Test that file paths are properly redacted."""
        data = CheckpointData(
            version=CURRENT_CHECKPOINT_VERSION,
            last_completed_stage="MATCH",
            voiceover_path="/home/user/projects/voiceover.srt",
            voiceover_hash="abc123",
            analyze={"keywords": ["test"]},
            video_search={"videos": [{"id": "vid1", "file_path": "/home/user/videos/test.mp4"}]},
            caption={},
            match={},
            iterative_match={},
            download_segments={},
            stage_metrics={},
        )

        checkpoint_mgr = CheckpointManager(temp_project_dir, config_hash="test_hash")
        checkpoint_mgr.data = data
        checkpoint_mgr.save("MATCH", data.to_dict())

        export_path = temp_project_dir / "redacted.json"
        success = checkpoint_mgr.export_checkpoint_redacted(str(export_path))

        assert success

        with open(export_path) as f:
            exported = json.load(f)

        # Check voiceover_path is redacted
        assert 'REDACTED' in str(exported.get('voiceover_path', ''))

    def test_redact_sensitive_fields(self, temp_project_dir):
        """Test that sensitive fields (api_key, token, etc.) are redacted."""
        data = CheckpointData(
            version=CURRENT_CHECKPOINT_VERSION,
            last_completed_stage="MATCH",
            voiceover_path="/path/voiceover.srt",
            voiceover_hash="abc123",
            analyze={"keywords": ["test"]},
            video_search={
                "api_key": "secret123",
                "auth_token": "token_abc",
                "videos": [{"id": "vid1"}],
            },
            caption={"cookie": "session=abc123"},
            match={},
            iterative_match={},
            download_segments={},
            stage_metrics={},
        )

        checkpoint_mgr = CheckpointManager(temp_project_dir, config_hash="test_hash")
        checkpoint_mgr.data = data
        checkpoint_mgr.save("MATCH", data.to_dict())

        export_path = temp_project_dir / "redacted.json"
        success = checkpoint_mgr.export_checkpoint_redacted(str(export_path))

        assert success

        with open(export_path) as f:
            exported = json.load(f)

        # Sensitive fields should be redacted
        vs = exported.get('video_search', {})
        assert vs.get('api_key') == '[REDACTED]' or 'REDACTED' in str(vs.get('api_key', ''))
        assert vs.get('auth_token') == '[REDACTED]' or 'REDACTED' in str(vs.get('auth_token', ''))

    def test_redaction_preserves_structure(self, temp_project_dir):
        """Test that redaction preserves stage structure and metrics."""
        data = CheckpointData(
            version=CURRENT_CHECKPOINT_VERSION,
            last_completed_stage="MATCH",
            voiceover_path="/path/voiceover.srt",
            voiceover_hash="abc123",
            analyze={"keywords": ["test"]},
            video_search={"videos": [{"id": "vid1", "title": "Test Video"}]},
            caption={"captions": {"vid1": {"text": "Hello"}}},
            match={"segments": [{"id": "seg1"}]},
            iterative_match={},
            download_segments={},
            stage_metrics={
                "VIDEO_SEARCH": {"duration": 1.5, "videos_found": 10},
                "CAPTION": {"duration": 2.0, "captions_fetched": 5},
            },
        )

        checkpoint_mgr = CheckpointManager(temp_project_dir, config_hash="test_hash")
        checkpoint_mgr.data = data
        checkpoint_mgr.save("MATCH", data.to_dict())

        export_path = temp_project_dir / "redacted.json"
        success = checkpoint_mgr.export_checkpoint_redacted(str(export_path))

        assert success

        with open(export_path) as f:
            exported = json.load(f)

        # Verify structure is preserved
        assert 'video_search' in exported
        assert 'caption' in exported
        assert 'match' in exported
        assert 'stage_metrics' in exported

        # Verify metrics are preserved
        assert 'VIDEO_SEARCH' in exported['stage_metrics']
        assert exported['stage_metrics']['VIDEO_SEARCH']['duration'] == 1.5

        # Verify export info indicates redaction
        assert exported['_export_info']['redacted'] is True

    def test_redaction_with_custom_patterns(self, temp_project_dir):
        """Test redaction with custom patterns."""
        data = CheckpointData(
            version=CURRENT_CHECKPOINT_VERSION,
            last_completed_stage="MATCH",
            voiceover_path="/path/voiceover.srt",
            voiceover_hash="abc123",
            analyze={"keywords": ["test"]},
            video_search={"videos": [{"id": "vid1", "custom_field": "sensitive_data"}]},
            caption={},
            match={},
            iterative_match={},
            download_segments={},
            stage_metrics={},
        )

        checkpoint_mgr = CheckpointManager(temp_project_dir, config_hash="test_hash")
        checkpoint_mgr.data = data
        checkpoint_mgr.save("MATCH", data.to_dict())

        # Custom patterns that redact 'custom_field'
        custom_patterns = {
            'redact_fields': ['custom_field'],
            'redact_urls': False,
            'redact_file_paths': False,
        }

        export_path = temp_project_dir / "redacted.json"
        success = checkpoint_mgr.export_checkpoint_redacted(
            str(export_path), redact_patterns=custom_patterns
        )

        assert success

        with open(export_path) as f:
            exported = json.load(f)

        # Custom field should be redacted
        videos = exported.get('video_search', {}).get('videos', [])
        if videos:
            assert 'custom_field' not in videos[0] or videos[0].get('custom_field') == '[REDACTED]'

    def test_redaction_empty_checkpoint(self, temp_project_dir):
        """Test redaction when no checkpoint exists."""
        checkpoint_mgr = CheckpointManager(temp_project_dir, config_hash="test_hash")

        export_path = temp_project_dir / "redacted.json"
        success = checkpoint_mgr.export_checkpoint_redacted(str(export_path))

        assert not success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
