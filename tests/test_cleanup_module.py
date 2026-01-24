"""
Tests for cleanup module.

US-003: Add cleanup module unit tests

Tests for src/cleanup/ covering:
- FileDeleter file and directory deletion
- AudioCleanupService cleanup operations
- Edge cases (non-existent files, permission errors)
"""

import pytest

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit

import sys
import os
import stat
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cleanup import FileDeleter, AudioCleanupService, AudioCleanupResult


class TestFileDeleterDeleteFile:
    """Tests for FileDeleter.delete_file()."""

    def test_delete_existing_file(self, tmp_path):
        """Test delete_file() removes specified file."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()

        deleter = FileDeleter()
        result = deleter.delete_file(test_file)

        assert result is True
        assert not test_file.exists()

    def test_delete_nonexistent_file(self, tmp_path):
        """Test delete_file() handles non-existent file gracefully."""
        nonexistent = tmp_path / "does_not_exist.txt"
        assert not nonexistent.exists()

        deleter = FileDeleter()
        result = deleter.delete_file(nonexistent)

        # Should return True (file doesn't exist = success)
        assert result is True

    def test_delete_file_permission_error(self, tmp_path):
        """Test delete_file() handles permission error with retries."""
        test_file = tmp_path / "locked.txt"
        test_file.write_text("locked content")

        deleter = FileDeleter(max_retries=2, backoff_base=0.01)

        # Mock unlink to raise PermissionError
        with patch.object(Path, 'unlink', side_effect=PermissionError("File locked")):
            result = deleter.delete_file(test_file)

        # Should return False after retries exhausted
        assert result is False

    def test_delete_file_os_error(self, tmp_path):
        """Test delete_file() handles OSError."""
        test_file = tmp_path / "problematic.txt"
        test_file.write_text("content")

        deleter = FileDeleter()

        with patch.object(Path, 'unlink', side_effect=OSError("Device error")):
            result = deleter.delete_file(test_file)

        assert result is False


class TestFileDeleterDeleteDirectory:
    """Tests for FileDeleter.delete_directory()."""

    def test_delete_empty_directory(self, tmp_path):
        """Test delete_directory() removes empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert empty_dir.exists()

        deleter = FileDeleter()
        result = deleter.delete_directory(empty_dir)

        assert result is True
        assert not empty_dir.exists()

    def test_delete_directory_with_files(self, tmp_path):
        """Test delete_directory() removes directory with contents."""
        dir_with_files = tmp_path / "has_files"
        dir_with_files.mkdir()
        (dir_with_files / "file1.txt").write_text("content1")
        (dir_with_files / "file2.txt").write_text("content2")
        assert dir_with_files.exists()

        deleter = FileDeleter()
        result = deleter.delete_directory(dir_with_files)

        assert result is True
        assert not dir_with_files.exists()

    def test_delete_nested_directory(self, tmp_path):
        """Test delete_directory() removes nested directories."""
        nested = tmp_path / "level1" / "level2" / "level3"
        nested.mkdir(parents=True)
        (nested / "deep_file.txt").write_text("deep content")

        deleter = FileDeleter()
        result = deleter.delete_directory(tmp_path / "level1")

        assert result is True
        assert not (tmp_path / "level1").exists()


class TestAudioCleanupResult:
    """Tests for AudioCleanupResult dataclass."""

    def test_default_values(self):
        """Test AudioCleanupResult has correct defaults."""
        result = AudioCleanupResult()

        assert result.files_deleted == 0
        assert result.directories_deleted == 0
        assert result.files_failed == []
        assert result.cache_entries_cleaned == 0
        assert result.dry_run is False

    def test_custom_values(self):
        """Test AudioCleanupResult accepts custom values."""
        failed_files = [Path("/test/file1.mp3"), Path("/test/file2.mp3")]
        result = AudioCleanupResult(
            files_deleted=5,
            directories_deleted=2,
            files_failed=failed_files,
            cache_entries_cleaned=3,
            dry_run=True
        )

        assert result.files_deleted == 5
        assert result.directories_deleted == 2
        assert len(result.files_failed) == 2
        assert result.cache_entries_cleaned == 3
        assert result.dry_run is True


class TestAudioCleanupServiceShouldCleanup:
    """Tests for AudioCleanupService.should_cleanup()."""

    def test_should_cleanup_enabled(self):
        """Test should_cleanup() returns True when enabled."""
        mock_config = Mock()
        mock_config.download.audio_first.delete_audio_after_video = True

        service = AudioCleanupService(
            file_deleter=Mock(),
            checkpoint=Mock(),
            config=mock_config
        )

        assert service.should_cleanup() is True

    def test_should_cleanup_disabled(self):
        """Test should_cleanup() returns False when disabled."""
        mock_config = Mock()
        mock_config.download.audio_first.delete_audio_after_video = False

        service = AudioCleanupService(
            file_deleter=Mock(),
            checkpoint=Mock(),
            config=mock_config
        )

        assert service.should_cleanup() is False

    def test_should_cleanup_no_audio_first_config(self):
        """Test should_cleanup() returns False when audio_first config missing."""
        mock_config = Mock()
        mock_config.download.audio_first = None

        service = AudioCleanupService(
            file_deleter=Mock(),
            checkpoint=Mock(),
            config=mock_config
        )

        assert service.should_cleanup() is False

    def test_should_cleanup_attribute_error(self):
        """Test should_cleanup() handles AttributeError gracefully."""
        mock_config = Mock(spec=[])  # Empty spec means no attributes

        service = AudioCleanupService(
            file_deleter=Mock(),
            checkpoint=Mock(),
            config=mock_config
        )

        assert service.should_cleanup() is False


class TestAudioCleanupServiceCleanup:
    """Tests for AudioCleanupService.cleanup()."""

    def test_cleanup_when_disabled(self):
        """Test cleanup() returns early when cleanup disabled."""
        mock_config = Mock()
        mock_config.download.audio_first.delete_audio_after_video = False

        service = AudioCleanupService(
            file_deleter=Mock(),
            checkpoint=Mock(),
            config=mock_config
        )

        mock_state = Mock()
        result = service.cleanup(mock_state)

        assert result.files_deleted == 0
        assert result.directories_deleted == 0

    def test_cleanup_no_audio_files(self):
        """Test cleanup() handles empty downloaded_audio list."""
        mock_config = Mock()
        mock_config.download.audio_first.delete_audio_after_video = True
        mock_config.downloaded_videos_dir = None

        service = AudioCleanupService(
            file_deleter=Mock(),
            checkpoint=Mock(),
            config=mock_config
        )

        mock_state = Mock()
        mock_state.downloaded_audio = []

        result = service.cleanup(mock_state)

        # Should complete without error
        assert result.files_deleted == 0

    def test_cleanup_dry_run(self, tmp_path):
        """Test cleanup() in dry_run mode doesn't delete files."""
        # Create test audio file
        audio_file = tmp_path / "test_audio.mp3"
        audio_file.write_text("audio content")

        mock_config = Mock()
        mock_config.download.audio_first.delete_audio_after_video = True
        mock_config.downloaded_videos_dir = str(tmp_path)

        mock_checkpoint = Mock()
        mock_checkpoint.get_stage_data.return_value = {}

        service = AudioCleanupService(
            file_deleter=FileDeleter(),
            checkpoint=mock_checkpoint,
            config=mock_config,
            dry_run=True
        )

        mock_state = Mock()
        mock_state.downloaded_audio = [Mock(file=str(audio_file))]

        result = service.cleanup(mock_state)

        # File should still exist in dry run mode
        assert audio_file.exists()
        assert result.dry_run is True

    def test_cleanup_deletes_files(self, tmp_path):
        """Test cleanup() deletes audio files."""
        # Create test audio file
        audio_file = tmp_path / "test_audio.mp3"
        audio_file.write_text("audio content")
        assert audio_file.exists()

        mock_config = Mock()
        mock_config.download.audio_first.delete_audio_after_video = True
        mock_config.downloaded_videos_dir = str(tmp_path)
        mock_config.cache_dir = str(tmp_path / "cache")

        mock_checkpoint = Mock()
        mock_checkpoint.get_stage_data.return_value = {}

        service = AudioCleanupService(
            file_deleter=FileDeleter(),
            checkpoint=mock_checkpoint,
            config=mock_config,
            dry_run=False
        )

        mock_state = Mock()
        mock_state.downloaded_audio = [Mock(file=str(audio_file))]

        # Patch print to avoid output during test
        with patch('builtins.print'):
            result = service.cleanup(mock_state)

        # File should be deleted
        assert not audio_file.exists()
        assert result.files_deleted == 1

    def test_cleanup_preserves_non_audio_files(self, tmp_path):
        """Test cleanup() only deletes specified audio files."""
        # Create test files
        audio_file = tmp_path / "to_delete.mp3"
        audio_file.write_text("audio")

        preserve_file = tmp_path / "keep_this.mp4"
        preserve_file.write_text("video")

        mock_config = Mock()
        mock_config.download.audio_first.delete_audio_after_video = True
        mock_config.downloaded_videos_dir = str(tmp_path)
        mock_config.cache_dir = str(tmp_path / "cache")

        mock_checkpoint = Mock()
        mock_checkpoint.get_stage_data.return_value = {}

        service = AudioCleanupService(
            file_deleter=FileDeleter(),
            checkpoint=mock_checkpoint,
            config=mock_config
        )

        mock_state = Mock()
        # Only audio_file is in the list
        mock_state.downloaded_audio = [Mock(file=str(audio_file))]

        with patch('builtins.print'):
            service.cleanup(mock_state)

        # Audio file deleted, other file preserved
        assert not audio_file.exists()
        assert preserve_file.exists()


class TestAudioCleanupServiceCollectFiles:
    """Tests for AudioCleanupService._collect_files()."""

    def test_collect_files_with_file_attribute(self, tmp_path):
        """Test _collect_files() handles objects with 'file' attribute."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_text("content")

        mock_config = Mock()
        service = AudioCleanupService(
            file_deleter=Mock(),
            checkpoint=Mock(),
            config=mock_config
        )

        mock_audio = Mock()
        mock_audio.file = str(audio_file)

        mock_state = Mock()
        mock_state.downloaded_audio = [mock_audio]

        files = service._collect_files(mock_state)

        assert len(files) == 1
        assert audio_file.resolve() in files

    def test_collect_files_with_dict(self, tmp_path):
        """Test _collect_files() handles dict entries with 'file' key."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_text("content")

        mock_config = Mock()
        service = AudioCleanupService(
            file_deleter=Mock(),
            checkpoint=Mock(),
            config=mock_config
        )

        mock_state = Mock()
        mock_state.downloaded_audio = [{'file': str(audio_file)}]

        files = service._collect_files(mock_state)

        assert len(files) == 1

    def test_collect_files_skips_nonexistent(self, tmp_path):
        """Test _collect_files() skips non-existent files."""
        mock_config = Mock()
        service = AudioCleanupService(
            file_deleter=Mock(),
            checkpoint=Mock(),
            config=mock_config
        )

        mock_state = Mock()
        mock_state.downloaded_audio = [
            Mock(file=str(tmp_path / "nonexistent.mp3"))
        ]

        files = service._collect_files(mock_state)

        assert len(files) == 0

    def test_collect_files_deduplicates(self, tmp_path):
        """Test _collect_files() deduplicates same file paths."""
        audio_file = tmp_path / "audio.mp3"
        audio_file.write_text("content")

        mock_config = Mock()
        service = AudioCleanupService(
            file_deleter=Mock(),
            checkpoint=Mock(),
            config=mock_config
        )

        mock_state = Mock()
        # Same file listed twice
        mock_state.downloaded_audio = [
            Mock(file=str(audio_file)),
            Mock(file=str(audio_file))
        ]

        files = service._collect_files(mock_state)

        # Should only have one entry
        assert len(files) == 1


class TestAudioCleanupServiceDeleteOrphanedDirectories:
    """Tests for AudioCleanupService._delete_orphaned_directories()."""

    def test_delete_orphaned_audio_directories(self, tmp_path):
        """Test _delete_orphaned_directories() removes *_audio dirs."""
        # Create audio directories matching tier patterns
        (tmp_path / "video1_s_audio").mkdir()
        (tmp_path / "video2_m_audio").mkdir()
        (tmp_path / "video3_l_audio").mkdir()
        # Create non-audio directory (should be preserved)
        (tmp_path / "video4_data").mkdir()

        mock_config = Mock()
        mock_config.downloaded_videos_dir = str(tmp_path)

        service = AudioCleanupService(
            file_deleter=FileDeleter(),
            checkpoint=Mock(),
            config=mock_config
        )

        deleted_count = service._delete_orphaned_directories()

        assert deleted_count == 3
        assert not (tmp_path / "video1_s_audio").exists()
        assert not (tmp_path / "video2_m_audio").exists()
        assert not (tmp_path / "video3_l_audio").exists()
        assert (tmp_path / "video4_data").exists()  # Preserved

    def test_delete_orphaned_no_video_dir(self):
        """Test _delete_orphaned_directories() handles missing video dir."""
        mock_config = Mock()
        mock_config.downloaded_videos_dir = None

        service = AudioCleanupService(
            file_deleter=Mock(),
            checkpoint=Mock(),
            config=mock_config
        )

        deleted_count = service._delete_orphaned_directories()

        assert deleted_count == 0

    def test_delete_orphaned_nonexistent_dir(self, tmp_path):
        """Test _delete_orphaned_directories() handles non-existent video dir."""
        mock_config = Mock()
        mock_config.downloaded_videos_dir = str(tmp_path / "nonexistent")

        service = AudioCleanupService(
            file_deleter=Mock(),
            checkpoint=Mock(),
            config=mock_config
        )

        deleted_count = service._delete_orphaned_directories()

        assert deleted_count == 0
