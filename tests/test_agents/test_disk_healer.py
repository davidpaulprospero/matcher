"""Unit tests for DiskHealer."""

import shutil
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agents.healers.disk import DiskHealer
from src.agents.base import HealerResult, HealerAction


class TestDiskHealerInit:
    """Test DiskHealer initialization."""

    def test_init_stores_config_and_project(self):
        """Test that DiskHealer stores config and project_dir."""
        config = MagicMock()
        healer = DiskHealer(config, "/path/to/project")

        assert healer.config is config
        assert healer.project_dir == "/path/to/project"

    def test_init_has_correct_name(self):
        """Test that DiskHealer has correct name."""
        config = MagicMock()
        healer = DiskHealer(config, "/tmp")

        assert healer.name == "disk-healer"

    def test_init_has_min_free_space_constant(self):
        """Test that DiskHealer has MIN_FREE_SPACE_GB constant."""
        config = MagicMock()
        healer = DiskHealer(config, "/tmp")

        assert hasattr(healer, 'MIN_FREE_SPACE_GB')
        assert healer.MIN_FREE_SPACE_GB >= 0.5  # At least 0.5 GB

    def test_init_has_cache_dirs_list(self):
        """Test that DiskHealer has CACHE_DIRS list."""
        config = MagicMock()
        healer = DiskHealer(config, "/tmp")

        assert hasattr(healer, 'CACHE_DIRS')
        assert isinstance(healer.CACHE_DIRS, list)
        assert len(healer.CACHE_DIRS) > 0


class TestDiskHealerCanHandle:
    """Test DiskHealer.can_handle() method."""

    def test_can_handle_disk_full(self):
        """Test that DiskHealer can handle disk full errors."""
        config = MagicMock()
        healer = DiskHealer(config, "/tmp")

        error = Exception("No space left on device")
        assert healer.can_handle(error, "OUTPUT") is True

        error = Exception("Disk full - cannot write file")
        assert healer.can_handle(error, "OUTPUT") is True

    def test_can_handle_errno_28(self):
        """Test that DiskHealer can handle ENOSPC (errno 28)."""
        config = MagicMock()
        healer = DiskHealer(config, "/tmp")

        error = OSError("[Errno 28] No space left on device")
        assert healer.can_handle(error, "OUTPUT") is True

    def test_can_handle_permission_denied(self):
        """Test that DiskHealer can handle permission denied errors."""
        config = MagicMock()
        healer = DiskHealer(config, "/tmp")

        error = PermissionError("Permission denied")
        assert healer.can_handle(error, "OUTPUT") is True

        error = Exception("Access denied: cannot write to directory")
        assert healer.can_handle(error, "OUTPUT") is True

    def test_can_handle_errno_13(self):
        """Test that DiskHealer can handle EACCES (errno 13)."""
        config = MagicMock()
        healer = DiskHealer(config, "/tmp")

        error = OSError("[Errno 13] Permission denied")
        assert healer.can_handle(error, "OUTPUT") is True

    def test_can_handle_storage_errors(self):
        """Test that DiskHealer can handle generic storage errors."""
        config = MagicMock()
        healer = DiskHealer(config, "/tmp")

        error = Exception("Storage error: disk is read-only")
        assert healer.can_handle(error, "DOWNLOAD") is True

    def test_cannot_handle_unrelated_error(self):
        """Test that DiskHealer doesn't handle unrelated errors."""
        config = MagicMock()
        healer = DiskHealer(config, "/tmp")

        error = Exception("Rate limit exceeded")
        assert healer.can_handle(error, "OUTPUT") is False

        error = Exception("Network timeout")
        assert healer.can_handle(error, "OUTPUT") is False


class TestDiskHealerHandleDiskFull:
    """Test DiskHealer._handle_disk_full() method."""

    def test_disk_full_cleans_project_caches(self, tmp_path):
        """Test that disk full handler cleans project caches."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create cache directories with files
        for cache_rel in DiskHealer.CACHE_DIRS:
            cache_path = project_dir / cache_rel
            cache_path.mkdir(parents=True, exist_ok=True)
            (cache_path / "test_file.json").write_text('{"test": true}')

        healer = DiskHealer(config, str(project_dir))
        state = MagicMock()

        result = healer._handle_disk_full(Exception("disk full"), state)

        assert result.success is True
        assert result.action == HealerAction.RETRY
        assert result.details.get('freed_bytes', 0) > 0

    def test_disk_full_returns_success_when_space_freed(self, tmp_path):
        """Test that disk full returns success when space is freed."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Create cache with content
        cache_path = project_dir / ".cache" / "llm_responses"
        cache_path.mkdir(parents=True, exist_ok=True)
        (cache_path / "response1.json").write_text('x' * 1000)

        healer = DiskHealer(config, str(project_dir))
        state = MagicMock()

        result = healer._handle_disk_full(Exception("no space"), state)

        assert result.success is True
        assert "cleaned_dirs" in result.details

    def test_disk_full_fails_when_no_space_freed(self, tmp_path):
        """Test that disk full fails when no space can be freed."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        # No cache directories to clean

        healer = DiskHealer(config, str(project_dir))
        state = MagicMock()

        # Mock _get_free_space to return very low value
        with patch.object(healer, '_get_free_space', return_value=0):
            result = healer._handle_disk_full(Exception("disk full"), state)

        assert result.success is False
        assert "disk" in result.message.lower()


class TestDiskHealerHandlePermission:
    """Test DiskHealer._handle_permission_error() method."""

    def test_permission_error_succeeds_when_dir_writable(self, tmp_path):
        """Test permission handler succeeds when project dir is writable."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        healer = DiskHealer(config, str(project_dir))
        state = MagicMock()

        result = healer._handle_permission_error(Exception("permission"), state)

        assert result.success is True
        assert result.action == HealerAction.RETRY

    def test_permission_error_fails_when_dir_not_writable(self, tmp_path):
        """Test permission handler fails when project dir is not writable."""
        config = MagicMock()

        healer = DiskHealer(config, "/nonexistent/path/that/does/not/exist")
        state = MagicMock()

        result = healer._handle_permission_error(Exception("permission"), state)

        assert result.success is False
        assert "permission" in result.message.lower()


class TestDiskHealerGetDirSize:
    """Test DiskHealer._get_dir_size() method."""

    def test_get_dir_size_returns_total_bytes(self, tmp_path):
        """Test that _get_dir_size returns correct total bytes."""
        config = MagicMock()
        healer = DiskHealer(config, str(tmp_path))

        # Create directory with known content
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("x" * 100)
        (test_dir / "file2.txt").write_text("y" * 200)

        size = healer._get_dir_size(test_dir)

        assert size == 300

    def test_get_dir_size_handles_empty_directory(self, tmp_path):
        """Test that _get_dir_size handles empty directory."""
        config = MagicMock()
        healer = DiskHealer(config, str(tmp_path))

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        size = healer._get_dir_size(empty_dir)

        assert size == 0

    def test_get_dir_size_handles_nonexistent_directory(self, tmp_path):
        """Test that _get_dir_size handles nonexistent directory."""
        config = MagicMock()
        healer = DiskHealer(config, str(tmp_path))

        nonexistent = tmp_path / "does_not_exist"

        size = healer._get_dir_size(nonexistent)

        assert size == 0


class TestDiskHealerGetFreeSpace:
    """Test DiskHealer._get_free_space() method."""

    def test_get_free_space_returns_bytes(self, tmp_path):
        """Test that _get_free_space returns free space in bytes."""
        config = MagicMock()
        healer = DiskHealer(config, str(tmp_path))

        free_space = healer._get_free_space(tmp_path)

        assert isinstance(free_space, int)
        assert free_space >= 0

    def test_get_free_space_handles_invalid_path(self, tmp_path):
        """Test that _get_free_space handles invalid paths gracefully."""
        config = MagicMock()
        healer = DiskHealer(config, str(tmp_path))

        nonexistent = Path("/this/path/definitely/does/not/exist/anywhere")

        # Should return 0 for invalid paths
        free_space = healer._get_free_space(nonexistent)

        assert free_space == 0


class TestDiskHealerFormatSize:
    """Test DiskHealer._format_size() method."""

    def test_format_size_bytes(self, tmp_path):
        """Test formatting small sizes as bytes."""
        config = MagicMock()
        healer = DiskHealer(config, str(tmp_path))

        result = healer._format_size(500)

        assert "500" in result
        assert "B" in result

    def test_format_size_kilobytes(self, tmp_path):
        """Test formatting as kilobytes."""
        config = MagicMock()
        healer = DiskHealer(config, str(tmp_path))

        result = healer._format_size(5 * 1024)

        assert "KB" in result

    def test_format_size_megabytes(self, tmp_path):
        """Test formatting as megabytes."""
        config = MagicMock()
        healer = DiskHealer(config, str(tmp_path))

        result = healer._format_size(50 * 1024 * 1024)

        assert "MB" in result

    def test_format_size_gigabytes(self, tmp_path):
        """Test formatting as gigabytes."""
        config = MagicMock()
        healer = DiskHealer(config, str(tmp_path))

        result = healer._format_size(5 * 1024 * 1024 * 1024)

        assert "GB" in result


class TestDiskHealerCheckSpace:
    """Test DiskHealer.check_space() method."""

    def test_check_space_returns_tuple(self, tmp_path):
        """Test that check_space returns (bool, float) tuple."""
        config = MagicMock()
        healer = DiskHealer(config, str(tmp_path))

        result = healer.check_space()

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)

    def test_check_space_with_custom_required_gb(self, tmp_path):
        """Test check_space with custom required_gb parameter."""
        config = MagicMock()
        healer = DiskHealer(config, str(tmp_path))

        # Very small requirement should pass
        has_enough, free_gb = healer.check_space(required_gb=0.0001)
        assert has_enough is True

    def test_check_space_uses_min_free_space_default(self, tmp_path):
        """Test that check_space uses MIN_FREE_SPACE_GB as default."""
        config = MagicMock()
        healer = DiskHealer(config, str(tmp_path))

        # Just verify it doesn't crash with no argument
        has_enough, free_gb = healer.check_space()

        assert free_gb >= 0


class TestDiskHealerFindOldFiles:
    """Test DiskHealer._find_old_files() method."""

    def test_find_old_files_returns_list(self, tmp_path):
        """Test that _find_old_files returns a list of Paths."""
        config = MagicMock()
        healer = DiskHealer(config, str(tmp_path))

        result = healer._find_old_files(tmp_path)

        assert isinstance(result, list)

    def test_find_old_files_empty_directory(self, tmp_path):
        """Test _find_old_files on empty directory."""
        config = MagicMock()
        healer = DiskHealer(config, str(tmp_path))

        result = healer._find_old_files(tmp_path, days=0)

        assert len(result) == 0

    def test_find_old_files_nonexistent_directory(self, tmp_path):
        """Test _find_old_files on nonexistent directory."""
        config = MagicMock()
        healer = DiskHealer(config, str(tmp_path))

        nonexistent = tmp_path / "does_not_exist"

        result = healer._find_old_files(nonexistent)

        assert result == []


class TestDiskHealerFix:
    """Test DiskHealer.fix() method routing."""

    def test_fix_routes_disk_full(self, tmp_path):
        """Test that fix() routes disk full errors correctly."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        healer = DiskHealer(config, str(project_dir))
        state = MagicMock()

        # Should try disk full handling
        with patch.object(healer, '_get_free_space', return_value=10 * 1024**3):
            result = healer.fix(Exception("No space left on device"), state, "OUTPUT")

        assert isinstance(result, HealerResult)

    def test_fix_routes_permission_error(self, tmp_path):
        """Test that fix() routes permission errors correctly."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        healer = DiskHealer(config, str(project_dir))
        state = MagicMock()

        result = healer.fix(Exception("Permission denied: /some/path"), state, "OUTPUT")

        assert isinstance(result, HealerResult)

    def test_fix_routes_errno_28(self, tmp_path):
        """Test that fix() routes ENOSPC errors correctly."""
        config = MagicMock()
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        healer = DiskHealer(config, str(project_dir))
        state = MagicMock()

        result = healer.fix(OSError("[Errno 28] No space"), state, "OUTPUT")

        assert isinstance(result, HealerResult)
