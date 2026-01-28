"""
Tests for transcript cache rebuild - ensures both 'source_file' and 'video_path' field names work.

This test prevents regressions of the bug where:
- Cached transcripts use 'source_file' field
- But _rebuild_text_metadata() was looking for 'video_path'
- Result: 0 entries rebuilt even with 300+ cached transcripts

The fix checks both field names: source_file (preferred) or video_path (legacy).
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.stages.transcribe import TranscribeStage
from src.state import PipelineState


class TestTranscriptCacheFieldNames:
    """Test that transcript cache rebuild handles different field names."""

    @pytest.mark.fast
    def test_rebuild_handles_source_file_field(self, tmp_path):
        """Rebuild should work with 'source_file' field (current format)."""
        # Create mock project structure
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        cache_dir = project_dir / ".cache" / "transcriptions"
        cache_dir.mkdir(parents=True)

        # Create checkpoint
        checkpoint_path = project_dir / "checkpoint.json"
        checkpoint_path.write_text("{}")

        # Create a cached transcript with 'source_file' field
        video_path = str(tmp_path / "video.mp4")
        transcript_data = [
            {
                "index": 1,
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "Hello world",
                "source_file": video_path  # Current format
            },
            {
                "index": 2,
                "start_time": 5.0,
                "end_time": 10.0,
                "text": "Test segment",
                "source_file": video_path
            }
        ]
        transcript_file = cache_dir / "abc123.json"
        transcript_file.write_text(json.dumps(transcript_data))

        # Create project config
        (project_dir / "project_config.yaml").write_text("""
project:
  name: test
cache:
  cache_dir: ".cache"
""")

        # Create mock state and checkpoint
        state = PipelineState()
        checkpoint = MagicMock()
        checkpoint.checkpoint_path = checkpoint_path

        # Run rebuild
        stage = TranscribeStage()
        stage._rebuild_text_metadata(state, checkpoint)

        # Should have found the transcripts
        assert len(state.transcripts) >= 1
        assert len(state.text_metadata) >= 2

    @pytest.mark.fast
    def test_rebuild_handles_video_path_field(self, tmp_path):
        """Rebuild should work with 'video_path' field (legacy format)."""
        # Create mock project structure
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        cache_dir = project_dir / ".cache" / "transcriptions"
        cache_dir.mkdir(parents=True)

        # Create checkpoint
        checkpoint_path = project_dir / "checkpoint.json"
        checkpoint_path.write_text("{}")

        # Create a cached transcript with 'video_path' field (legacy)
        video_path = str(tmp_path / "video.mp4")
        transcript_data = [
            {
                "index": 1,
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "Hello world",
                "video_path": video_path  # Legacy format
            }
        ]
        transcript_file = cache_dir / "abc123.json"
        transcript_file.write_text(json.dumps(transcript_data))

        # Create project config
        (project_dir / "project_config.yaml").write_text("""
project:
  name: test
cache:
  cache_dir: ".cache"
""")

        # Create mock state and checkpoint
        state = PipelineState()
        checkpoint = MagicMock()
        checkpoint.checkpoint_path = checkpoint_path

        # Run rebuild
        stage = TranscribeStage()
        stage._rebuild_text_metadata(state, checkpoint)

        # Should have found the transcripts
        assert len(state.transcripts) >= 1
        assert len(state.text_metadata) >= 1

    @pytest.mark.fast
    def test_rebuild_prefers_source_file_over_video_path(self, tmp_path):
        """When both fields exist, source_file should be used."""
        # Create mock project structure
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        cache_dir = project_dir / ".cache" / "transcriptions"
        cache_dir.mkdir(parents=True)

        # Create checkpoint
        checkpoint_path = project_dir / "checkpoint.json"
        checkpoint_path.write_text("{}")

        # Create a cached transcript with BOTH fields
        source_path = str(tmp_path / "source_video.mp4")
        legacy_path = str(tmp_path / "legacy_video.mp4")
        transcript_data = [
            {
                "index": 1,
                "start_time": 0.0,
                "end_time": 5.0,
                "text": "Hello world",
                "source_file": source_path,  # Should be used
                "video_path": legacy_path     # Should be ignored
            }
        ]
        transcript_file = cache_dir / "abc123.json"
        transcript_file.write_text(json.dumps(transcript_data))

        # Create project config
        (project_dir / "project_config.yaml").write_text("""
project:
  name: test
cache:
  cache_dir: ".cache"
""")

        # Create mock state and checkpoint
        state = PipelineState()
        checkpoint = MagicMock()
        checkpoint.checkpoint_path = checkpoint_path

        # Run rebuild
        stage = TranscribeStage()
        stage._rebuild_text_metadata(state, checkpoint)

        # Should use source_file path
        assert source_path in state.transcripts
        assert legacy_path not in state.transcripts


class TestSceneDetectionVideoSources:
    """Test that scene detection finds videos from multiple sources."""

    @pytest.mark.fast
    def test_get_video_files_from_transcripts(self, tmp_path):
        """_get_video_files should find videos from state.transcripts."""
        from src.stages.scene_detection import SceneDetectionStage

        # Create a video file
        video_file = tmp_path / "video.mp4"
        video_file.write_text("fake video")

        # Create state with transcripts but no downloaded_videos
        state = PipelineState()
        state.downloaded_videos = []
        state.transcripts = {str(video_file): [{"text": "test"}]}

        stage = SceneDetectionStage()
        video_files = stage._get_video_files(state)

        assert len(video_files) == 1
        assert video_files[0] == video_file

    @pytest.mark.fast
    def test_get_video_files_from_remix_files(self, tmp_path):
        """_get_video_files should find videos from state.remix_files."""
        from src.stages.scene_detection import SceneDetectionStage

        # Create a video file
        video_file = tmp_path / "video.mp4"
        video_file.write_text("fake video")

        # Create state with remix_files but no downloaded_videos or transcripts
        state = PipelineState()
        state.downloaded_videos = []
        state.transcripts = {}
        state.remix_files = [str(video_file)]

        stage = SceneDetectionStage()
        video_files = stage._get_video_files(state)

        assert len(video_files) == 1
        assert video_files[0] == video_file

    @pytest.mark.fast
    def test_validate_inputs_accepts_transcripts(self, tmp_path):
        """validate_inputs should pass if state.transcripts has data."""
        from src.stages.scene_detection import SceneDetectionStage
        from src.config.base import Config

        state = PipelineState()
        state.downloaded_videos = []
        state.downloaded_audio = []
        state.transcripts = {"video.mp4": [{"text": "test"}]}
        state.remix_files = []

        config = Config()
        stage = SceneDetectionStage()
        result = stage.validate_inputs(state, config)

        assert result is None  # None means validation passed

    @pytest.mark.fast
    def test_validate_inputs_fails_when_all_empty(self):
        """validate_inputs should fail if all video sources are empty."""
        from src.stages.scene_detection import SceneDetectionStage
        from src.config.base import Config

        state = PipelineState()
        state.downloaded_videos = []
        state.downloaded_audio = []
        state.transcripts = {}
        state.remix_files = []

        config = Config()
        stage = SceneDetectionStage()
        result = stage.validate_inputs(state, config)

        assert result is not None
        assert "No videos" in result
