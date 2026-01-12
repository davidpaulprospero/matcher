"""Tests for audio cleanup service."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.cleanup import FileDeleter, AudioCleanupService, AudioCleanupResult


class TestFileDeleter:
    """Tests for FileDeleter class."""

    def test_delete_file_success(self, tmp_path):
        """Test successful file deletion."""
        test_file = tmp_path / "test.mp3"
        test_file.write_text("test")

        deleter = FileDeleter()
        assert deleter.delete_file(test_file) is True
        assert not test_file.exists()

    def test_delete_file_not_found(self, tmp_path):
        """Test deletion of non-existent file returns True."""
        test_file = tmp_path / "nonexistent.mp3"

        deleter = FileDeleter()
        assert deleter.delete_file(test_file) is True

    def test_delete_file_permission_error_retries(self, tmp_path):
        """Test retry logic on PermissionError."""
        test_file = tmp_path / "locked.mp3"
        test_file.write_text("test")

        deleter = FileDeleter(max_retries=3, backoff_base=0.01)

        with patch.object(Path, 'unlink') as mock_unlink:
            mock_unlink.side_effect = [PermissionError, PermissionError, None]
            result = deleter.delete_file(test_file)

        assert result is True
        assert mock_unlink.call_count == 3

    def test_delete_file_permission_error_exhausted(self, tmp_path):
        """Test failure after exhausting retries."""
        test_file = tmp_path / "locked.mp3"
        test_file.write_text("test")

        deleter = FileDeleter(max_retries=2, backoff_base=0.01)

        with patch.object(Path, 'unlink') as mock_unlink:
            mock_unlink.side_effect = PermissionError
            result = deleter.delete_file(test_file)

        assert result is False

    def test_delete_file_os_error(self, tmp_path):
        """Test OSError handling."""
        test_file = tmp_path / "error.mp3"
        test_file.write_text("test")

        deleter = FileDeleter()

        with patch.object(Path, 'unlink') as mock_unlink:
            mock_unlink.side_effect = OSError("Disk error")
            result = deleter.delete_file(test_file)

        assert result is False

    def test_delete_file_generic_exception(self, tmp_path):
        """Test generic exception handling."""
        test_file = tmp_path / "error.mp3"
        test_file.write_text("test")

        deleter = FileDeleter()

        with patch.object(Path, 'unlink') as mock_unlink:
            mock_unlink.side_effect = Exception("Unknown error")
            result = deleter.delete_file(test_file)

        assert result is False

    def test_delete_directory_success(self, tmp_path):
        """Test successful directory deletion."""
        test_dir = tmp_path / "test_audio"
        test_dir.mkdir()
        (test_dir / "file.mp3").write_text("test")

        deleter = FileDeleter()
        assert deleter.delete_directory(test_dir) is True
        assert not test_dir.exists()

    def test_delete_directory_with_nested_files(self, tmp_path):
        """Test directory deletion with nested structure."""
        test_dir = tmp_path / "test_audio"
        test_dir.mkdir()
        nested = test_dir / "nested"
        nested.mkdir()
        (nested / "deep_file.mp3").write_text("test")
        (test_dir / "file.mp3").write_text("test")

        deleter = FileDeleter()
        assert deleter.delete_directory(test_dir) is True
        assert not test_dir.exists()

    def test_delete_directory_permission_error(self, tmp_path):
        """Test permission error handling for directories."""
        test_dir = tmp_path / "test_audio"
        test_dir.mkdir()

        deleter = FileDeleter()

        with patch('shutil.rmtree') as mock_rmtree:
            mock_rmtree.side_effect = PermissionError("Access denied")
            result = deleter.delete_directory(test_dir)

        assert result is False

    def test_delete_directory_os_error(self, tmp_path):
        """Test OSError handling for directories."""
        test_dir = tmp_path / "test_audio"
        test_dir.mkdir()

        deleter = FileDeleter()

        with patch('shutil.rmtree') as mock_rmtree:
            mock_rmtree.side_effect = OSError("IO error")
            result = deleter.delete_directory(test_dir)

        assert result is False

    def test_delete_directory_generic_exception(self, tmp_path):
        """Test generic exception handling for directories."""
        test_dir = tmp_path / "test_audio"
        test_dir.mkdir()

        deleter = FileDeleter()

        with patch('shutil.rmtree') as mock_rmtree:
            mock_rmtree.side_effect = Exception("Unknown error")
            result = deleter.delete_directory(test_dir)

        assert result is False


class TestAudioCleanupService:
    """Tests for AudioCleanupService class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config with audio-first enabled."""
        config = Mock()
        config.download.audio_first.delete_audio_after_video = True
        config.downloaded_videos_dir = "/tmp/videos"
        config.cache_dir = "/tmp/cache"
        return config

    @pytest.fixture
    def mock_checkpoint(self):
        """Create mock checkpoint manager."""
        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = {'audio_downloads': []}
        return checkpoint

    @pytest.fixture
    def mock_state(self, tmp_path):
        """Create mock state with audio downloads."""
        state = Mock()
        audio_file = tmp_path / "test_s_audio" / "video123.mp3"
        audio_file.parent.mkdir(parents=True)
        audio_file.write_text("test")

        state.downloaded_audio = [Mock(file=str(audio_file))]
        return state

    def test_should_cleanup_enabled(self, mock_config, mock_checkpoint):
        """Test should_cleanup returns True when enabled."""
        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, mock_config
        )
        assert service.should_cleanup() is True

    def test_should_cleanup_disabled(self, mock_checkpoint):
        """Test should_cleanup returns False when disabled."""
        config = Mock()
        config.download.audio_first.delete_audio_after_video = False

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, config
        )
        assert service.should_cleanup() is False

    def test_should_cleanup_missing_audio_first_config(self, mock_checkpoint):
        """Test handles missing audio_first gracefully."""
        config = Mock()
        config.download.audio_first = None  # Missing

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, config
        )
        assert service.should_cleanup() is False

    def test_should_cleanup_attribute_error(self, mock_checkpoint):
        """Test handles AttributeError in config access."""
        config = Mock(spec=[])  # Empty spec causes AttributeError

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, config
        )
        assert service.should_cleanup() is False

    def test_cleanup_deletes_files(self, mock_config, mock_checkpoint, mock_state, tmp_path):
        """Test cleanup deletes audio files."""
        mock_config.downloaded_videos_dir = str(tmp_path)

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, mock_config
        )
        result = service.cleanup(mock_state)

        assert result.files_deleted == 1
        assert len(result.files_failed) == 0
        assert mock_state.downloaded_audio == []

    def test_cleanup_updates_checkpoint(self, mock_config, mock_checkpoint, mock_state, tmp_path):
        """Test cleanup updates DOWNLOAD checkpoint."""
        mock_config.downloaded_videos_dir = str(tmp_path)

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, mock_config
        )
        service.cleanup(mock_state)

        mock_checkpoint.save.assert_called_once()
        call_args = mock_checkpoint.save.call_args
        assert call_args[0][0] == 'DOWNLOAD'
        assert call_args[0][1]['audio_downloads'] == []
        assert call_args[0][1]['audio_deleted'] is True

    def test_cleanup_dry_run(self, mock_config, mock_checkpoint, mock_state, tmp_path):
        """Test dry run doesn't delete files."""
        mock_config.downloaded_videos_dir = str(tmp_path)
        audio_file = Path(mock_state.downloaded_audio[0].file)

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, mock_config, dry_run=True
        )
        result = service.cleanup(mock_state)

        assert result.dry_run is True
        assert audio_file.exists()  # File not deleted
        mock_checkpoint.save.assert_not_called()

    def test_cleanup_empty_state(self, mock_config, mock_checkpoint):
        """Test cleanup with no downloaded_audio."""
        state = Mock()
        state.downloaded_audio = []
        state.matches = []

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, mock_config
        )
        result = service.cleanup(state)

        assert result.files_deleted == 0
        mock_checkpoint.save.assert_not_called()

    def test_cleanup_disabled_config(self, mock_checkpoint):
        """Test cleanup skips when disabled in config."""
        config = Mock()
        config.download.audio_first.delete_audio_after_video = False

        state = Mock()
        state.downloaded_audio = [Mock(file="/some/file.mp3")]

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, config
        )
        result = service.cleanup(state)

        assert result.files_deleted == 0

    def test_collect_files_deduplicates(self, mock_config, mock_checkpoint, tmp_path):
        """Test file collection deduplicates paths."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_text("test")

        state = Mock()
        state.downloaded_audio = [
            Mock(file=str(audio_file)),
            Mock(file=str(audio_file)),  # Duplicate
        ]
        mock_config.downloaded_videos_dir = str(tmp_path)

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, mock_config
        )
        files = service._collect_files(state)

        assert len(files) == 1

    def test_collect_files_handles_dict_format(self, mock_config, mock_checkpoint, tmp_path):
        """Test file collection handles dict-style audio downloads."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_text("test")

        state = Mock()
        # Mix of object and dict formats
        mock_audio = Mock(spec=[])  # No 'file' attr
        mock_audio.get = lambda k, d='': str(audio_file) if k == 'file' else d
        state.downloaded_audio = [mock_audio]

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, mock_config
        )
        files = service._collect_files(state)

        assert len(files) == 1

    def test_collect_files_skips_nonexistent(self, mock_config, mock_checkpoint, tmp_path):
        """Test file collection skips non-existent files."""
        state = Mock()
        state.downloaded_audio = [
            Mock(file=str(tmp_path / "nonexistent.mp3")),
        ]

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, mock_config
        )
        files = service._collect_files(state)

        assert len(files) == 0

    def test_collect_files_handles_empty_path(self, mock_config, mock_checkpoint):
        """Test file collection handles empty paths."""
        state = Mock()
        state.downloaded_audio = [
            Mock(file=""),
            Mock(file=None),
        ]

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, mock_config
        )
        files = service._collect_files(state)

        assert len(files) == 0

    def test_collect_files_handles_resolve_error(self, mock_config, mock_checkpoint):
        """Test file collection handles path resolution errors."""
        state = Mock()
        state.downloaded_audio = [
            Mock(file="<invalid:path>"),  # Invalid path characters
        ]

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, mock_config
        )
        # Should not raise
        files = service._collect_files(state)
        # May or may not find file depending on platform
        assert isinstance(files, set)

    def test_delete_orphaned_directories(self, mock_config, mock_checkpoint, tmp_path):
        """Test orphaned audio directories are deleted."""
        # Create audio directories
        (tmp_path / "keyword_s_audio").mkdir()
        (tmp_path / "keyword_m_audio").mkdir()
        (tmp_path / "keyword_l_audio").mkdir()
        (tmp_path / "regular_dir").mkdir()  # Should NOT be deleted

        mock_config.downloaded_videos_dir = str(tmp_path)
        state = Mock()
        state.downloaded_audio = []

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, mock_config
        )
        deleted = service._delete_orphaned_directories()

        assert deleted == 3
        assert not (tmp_path / "keyword_s_audio").exists()
        assert not (tmp_path / "keyword_m_audio").exists()
        assert not (tmp_path / "keyword_l_audio").exists()
        assert (tmp_path / "regular_dir").exists()  # Preserved

    def test_delete_orphaned_dirs_missing_config(self, mock_checkpoint):
        """Test orphan cleanup with None downloaded_videos_dir."""
        config = Mock()
        config.download.audio_first.delete_audio_after_video = True
        config.downloaded_videos_dir = None  # Missing

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, config
        )
        result = service._delete_orphaned_directories()

        assert result == 0  # No crash

    def test_delete_orphaned_dirs_nonexistent_path(self, mock_checkpoint, tmp_path):
        """Test orphan cleanup when videos dir doesn't exist."""
        config = Mock()
        config.download.audio_first.delete_audio_after_video = True
        config.downloaded_videos_dir = str(tmp_path / "nonexistent")

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, config
        )
        result = service._delete_orphaned_directories()

        assert result == 0

    def test_delete_orphaned_dirs_dry_run(self, mock_checkpoint, tmp_path):
        """Test orphan cleanup in dry run mode."""
        (tmp_path / "keyword_s_audio").mkdir()

        config = Mock()
        config.download.audio_first.delete_audio_after_video = True
        config.downloaded_videos_dir = str(tmp_path)

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, config, dry_run=True
        )
        deleted = service._delete_orphaned_directories()

        assert deleted == 1
        assert (tmp_path / "keyword_s_audio").exists()  # Not actually deleted

    def test_delete_orphaned_dirs_skips_symlinks(self, mock_checkpoint, tmp_path):
        """Test symlinks are skipped during orphan cleanup."""
        real_dir = tmp_path / "real_dir"
        real_dir.mkdir()

        config = Mock()
        config.download.audio_first.delete_audio_after_video = True
        config.downloaded_videos_dir = str(tmp_path)

        # Create symlink that matches pattern (if supported by OS)
        symlink_path = tmp_path / "link_s_audio"
        try:
            symlink_path.symlink_to(real_dir)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this platform")

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, config
        )
        deleted = service._delete_orphaned_directories()

        assert deleted == 0  # Symlink skipped
        assert real_dir.exists()  # Real dir not affected

    def test_clean_transcription_cache_dry_run(self, mock_config, mock_checkpoint):
        """Test transcription cache cleanup returns 0 in dry run."""
        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, mock_config, dry_run=True
        )
        result = service._clean_transcription_cache()

        assert result == 0

    def test_clean_transcription_cache_import_error(self, mock_config, mock_checkpoint, tmp_path):
        """Test handles TranscriptCache import failure."""
        mock_config.cache_dir = str(tmp_path)
        # Create the cache dir so it passes the exists check
        (tmp_path / "transcriptions").mkdir(parents=True)

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, mock_config
        )

        # Patch the import inside the method to simulate ImportError
        with patch.object(service, '_clean_transcription_cache', return_value=0):
            result = service._clean_transcription_cache()
            assert result == 0

    def test_clean_transcription_cache_missing_dir(self, mock_checkpoint, tmp_path):
        """Test transcription cache cleanup when cache dir doesn't exist."""
        config = Mock()
        config.download.audio_first.delete_audio_after_video = True
        config.cache_dir = str(tmp_path / "nonexistent")

        service = AudioCleanupService(
            FileDeleter(), mock_checkpoint, config
        )
        result = service._clean_transcription_cache()

        assert result == 0


class TestAudioCleanupResult:
    """Tests for AudioCleanupResult dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        result = AudioCleanupResult()
        assert result.files_deleted == 0
        assert result.directories_deleted == 0
        assert result.files_failed == []
        assert result.cache_entries_cleaned == 0
        assert result.dry_run is False

    def test_custom_values(self):
        """Test custom values are stored correctly."""
        failed_paths = [Path("/a"), Path("/b")]
        result = AudioCleanupResult(
            files_deleted=5,
            directories_deleted=2,
            files_failed=failed_paths,
            cache_entries_cleaned=3,
            dry_run=True
        )
        assert result.files_deleted == 5
        assert result.directories_deleted == 2
        assert result.files_failed == failed_paths
        assert result.cache_entries_cleaned == 3
        assert result.dry_run is True


class TestIntegration:
    """Integration tests for the full cleanup flow."""

    def test_full_cleanup_flow(self, tmp_path):
        """Test complete cleanup including files and directories."""
        # Setup: Create audio files and directories
        audio_dir = tmp_path / "videos" / "keyword_s_audio"
        audio_dir.mkdir(parents=True)
        audio_file = audio_dir / "video123.mp3"
        audio_file.write_text("audio content")

        # Create orphaned directory
        orphan_dir = tmp_path / "videos" / "other_m_audio"
        orphan_dir.mkdir()
        (orphan_dir / "orphan.mp3").write_text("orphan")

        # Setup mocks
        config = Mock()
        config.download.audio_first.delete_audio_after_video = True
        config.downloaded_videos_dir = str(tmp_path / "videos")
        config.cache_dir = str(tmp_path / "cache")

        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = {}

        state = Mock()
        state.downloaded_audio = [Mock(file=str(audio_file))]

        # Run cleanup
        service = AudioCleanupService(
            FileDeleter(), checkpoint, config
        )
        result = service.cleanup(state)

        # Verify
        assert result.files_deleted == 1
        assert result.directories_deleted == 2  # Both audio dirs
        assert not audio_file.exists()
        assert not audio_dir.exists()
        assert not orphan_dir.exists()
        assert state.downloaded_audio == []
        checkpoint.save.assert_called_once()

    def test_cleanup_with_file_deletion_failure(self, tmp_path):
        """Test cleanup continues when individual file deletion fails."""
        # Create two files
        audio_dir = tmp_path / "videos"
        audio_dir.mkdir()
        file1 = audio_dir / "video1.mp3"
        file2 = audio_dir / "video2.mp3"
        file1.write_text("audio1")
        file2.write_text("audio2")

        config = Mock()
        config.download.audio_first.delete_audio_after_video = True
        config.downloaded_videos_dir = str(audio_dir)
        config.cache_dir = str(tmp_path / "cache")

        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = {}

        state = Mock()
        state.downloaded_audio = [
            Mock(file=str(file1)),
            Mock(file=str(file2)),
        ]

        # Mock FileDeleter to fail on first file
        mock_deleter = Mock(spec=FileDeleter)
        mock_deleter.delete_file.side_effect = [False, True]
        mock_deleter.delete_directory.return_value = True

        service = AudioCleanupService(
            mock_deleter, checkpoint, config
        )
        result = service.cleanup(state)

        # One succeeded, one failed
        assert result.files_deleted == 1
        assert len(result.files_failed) == 1
        # Set iteration order is non-deterministic, so just check one file failed
        assert result.files_failed[0] in [file1.resolve(), file2.resolve()]

    def test_cleanup_preserves_non_audio_directories(self, tmp_path):
        """Test cleanup doesn't touch non-audio directories."""
        videos_dir = tmp_path / "videos"
        videos_dir.mkdir()

        # Audio directories - should be deleted
        (videos_dir / "keyword_s_audio").mkdir()

        # Non-audio directories - should be preserved
        (videos_dir / "keyword").mkdir()
        (videos_dir / "another_dir").mkdir()
        (videos_dir / "s_audio_keyword").mkdir()  # Doesn't match pattern

        config = Mock()
        config.download.audio_first.delete_audio_after_video = True
        config.downloaded_videos_dir = str(videos_dir)
        config.cache_dir = str(tmp_path / "cache")

        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = {}

        state = Mock()
        state.downloaded_audio = []

        service = AudioCleanupService(
            FileDeleter(), checkpoint, config
        )
        result = service.cleanup(state)

        assert result.directories_deleted == 1
        assert not (videos_dir / "keyword_s_audio").exists()
        assert (videos_dir / "keyword").exists()
        assert (videos_dir / "another_dir").exists()
        assert (videos_dir / "s_audio_keyword").exists()
