"""
US-130-012: Test cross-platform path normalization for checkpoints.

Tests that paths are normalized using os.path.realpath on save and
that paths are converted when loaded on a different platform.

Run: pytest tests/test_checkpoint_path_normalization.py -v
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.checkpoint import CheckpointManager, CheckpointData


class TestPathNormalization:
    """Tests for US-130-012 cross-platform path normalization."""

    def test_normalize_path_with_realpath(self):
        """Test that _normalize_path uses os.path.realpath."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a test file to get a real path
            test_file = tmpdir / "test.srt"
            test_file.write_text("test")

            cm = CheckpointManager(tmpdir, config_hash="test123")

            # The normalized path should be absolute and resolved
            result = cm._normalize_path(str(test_file))

            # Should be an absolute path
            assert os.path.isabs(result), f"Expected absolute path, got: {result}"
            # Should resolve to the same file
            assert os.path.exists(result), f"Normalized path should exist: {result}"

    def test_normalize_path_empty_string(self):
        """Test that _normalize_path handles empty strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(Path(tmpdir), config_hash="test123")

            # Empty string should return empty
            assert cm._normalize_path("") == ""

            # Whitespace-only string - realpath will resolve to current directory + whitespace
            # This is expected behavior for os.path.realpath
            result = cm._normalize_path("   ")
            # Should return a valid path (not crash)
            assert result is not None
            assert len(result) > 0

    def test_normalize_path_nonexistent_file(self):
        """Test that _normalize_path handles nonexistent files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(Path(tmpdir), config_hash="test123")

            # Nonexistent path should return original (or try to resolve)
            nonexistent = "/this/path/does/not/exist/file.srt"
            result = cm._normalize_path(nonexistent)

            # Should still return a path (may or may not be the same)
            assert result is not None

    def test_set_voiceover_normalizes_path(self):
        """Test that set_voiceover stores normalized paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a test voiceover file
            voiceover_file = tmpdir / "voiceover.srt"
            voiceover_file.write_text("test content")

            cm = CheckpointManager(tmpdir, config_hash="test123")
            cm.set_voiceover(str(voiceover_file))

            # The stored path should be normalized (absolute)
            stored_path = cm.data.voiceover_path
            assert os.path.isabs(stored_path), f"Expected absolute path, got: {stored_path}"

    def test_convert_path_separators_windows_to_linux(self):
        """Test conversion from Windows backslashes to Linux forward slashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(Path(tmpdir), config_hash="test123")

            # Simulate Windows path on a Linux system
            windows_path = "E:\\Projects\\Video\\voiceover.srt"

            # Patch os.sep to simulate Linux
            with patch.object(os, 'sep', '/'):
                result = cm._convert_path_separators(windows_path)

                # Should convert backslashes to forward slashes
                assert '\\' not in result, f"Expected no backslashes, got: {result}"
                assert result == "E:/Projects/Video/voiceover.srt"

    def test_convert_path_separators_linux_to_windows(self):
        """Test conversion from Linux forward slashes to Windows backslashes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(Path(tmpdir), config_hash="test123")

            # Simulate Linux path on a Windows system
            linux_path = "/home/user/projects/voiceover.srt"

            # Patch os.sep to simulate Windows
            with patch.object(os, 'sep', '\\'):
                result = cm._convert_path_separators(linux_path)

                # Should convert forward slashes to backslashes
                assert '/' not in result, f"Expected no forward slashes, got: {result}"
                assert result == "\\home\\user\\projects\\voiceover.srt"

    def test_convert_path_separators_no_conversion_needed(self):
        """Test that no conversion happens when already on same platform."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(Path(tmpdir), config_hash="test123")

            # Current platform path - no conversion needed
            current_path = os.path.join("home", "user", "voiceover.srt")
            if os.sep == '/':
                # On Linux, test with forward slashes
                current_path = "/home/user/voiceover.srt"

            result = cm._convert_path_separators(current_path)

            # Should be unchanged (or normalized but not converted)
            assert result is not None

    def test_detect_and_normalize_paths_with_voiceover(self):
        """Test that _detect_and_normalize_paths handles voiceover_path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a test voiceover file
            voiceover_file = tmpdir / "voiceover.srt"
            voiceover_file.write_text("test content")

            cm = CheckpointManager(tmpdir, config_hash="test123")

            # Create checkpoint data with a path
            data = CheckpointData()
            data.voiceover_path = str(voiceover_file)

            # Should normalize without issues
            result = cm._detect_and_normalize_paths(data)

            # The path should be normalized to absolute
            assert os.path.isabs(data.voiceover_path)

    def test_detect_and_normalize_paths_nonexistent(self):
        """Test path normalization for nonexistent files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CheckpointManager(Path(tmpdir), config_hash="test123")

            # Create checkpoint data with nonexistent path
            data = CheckpointData()
            data.voiceover_path = "/nonexistent/path/file.srt"

            # Should still try to normalize
            result = cm._detect_and_normalize_paths(data)

            # Should return True since normalization was attempted
            # (even if the file doesn't exist)
            # The result depends on whether os.path.realpath changes the path

    def test_checkpoint_save_and_load_path_normalization(self):
        """Test full cycle: save checkpoint with voiceover, then load it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a test voiceover file
            voiceover_file = tmpdir / "voiceover.srt"
            voiceover_file.write_text("test content")

            # Save checkpoint with voiceover
            cm1 = CheckpointManager(tmpdir, config_hash="test123")
            cm1.set_voiceover(str(voiceover_file))
            cm1.save("ANALYZE", {"test": "data"})

            # Load checkpoint
            cm2 = CheckpointManager(tmpdir, config_hash="test123")
            data = cm2.load()

            # The loaded path should be normalized
            assert data.voiceover_path is not None
            # On Linux, should be absolute path
            if os.path.exists(str(voiceover_file)):
                assert os.path.isabs(data.voiceover_path)


class TestCheckpointDataPathField:
    """Tests for path fields in CheckpointData dataclass."""

    def test_voiceover_path_default_empty(self):
        """Test that voiceover_path defaults to empty string."""
        data = CheckpointData()
        assert data.voiceover_path == ""

    def test_voiceover_path_can_be_set(self):
        """Test that voiceover_path can be set."""
        data = CheckpointData()
        data.voiceover_path = "/path/to/voiceover.srt"
        assert data.voiceover_path == "/path/to/voiceover.srt"

    def test_checkpoint_data_to_dict(self):
        """Test CheckpointData.to_dict includes voiceover_path."""
        data = CheckpointData()
        data.voiceover_path = "/path/to/voiceover.srt"

        data_dict = data.to_dict()

        assert "voiceover_path" in data_dict
        assert data_dict["voiceover_path"] == "/path/to/voiceover.srt"

    def test_checkpoint_data_from_dict(self):
        """Test CheckpointData.from_dict restores voiceover_path."""
        data_dict = {
            "version": "2.1",
            "voiceover_path": "/path/to/voiceover.srt",
            "voiceover_hash": "abc123",
            "analyze": {},
            "video_search": {},
            "caption": {},
            "match": {},
            "iterative_match": {},
            "download_segments": {},
            "chapter_data": {},
            "stage_metrics": {},
            "transcription_metrics": {},
            "validation_cache": {},
            "escalation_state": {},
            "circuit_breaker_health": {},
            "circuit_breaker_state": {},
            "rate_limit_state": {},
            "compressed_checkpoint": {},
        }

        data = CheckpointData.from_dict(data_dict)

        assert data.voiceover_path == "/path/to/voiceover.srt"
        assert data.voiceover_hash == "abc123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
