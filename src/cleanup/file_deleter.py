"""Low-level file deletion with Windows retry logic."""
import os
import stat
import time
import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FileDeleter:
    """File deletion with retry for Windows file locking.

    Handles common edge cases:
    - Windows file locking (retries with exponential backoff)
    - Read-only files (attempts to remove readonly flag)
    - Already deleted files (returns success)
    - Permission errors (logs warning, returns failure)
    - Dry-run mode for previewing deletions
    - Batch deletion for processing multiple files
    """

    def __init__(self, max_retries: int = 3, backoff_base: float = 0.5, dry_run: bool = False):
        """Initialize FileDeleter.

        Args:
            max_retries: Maximum number of retry attempts for locked files
            backoff_base: Base time in seconds for exponential backoff
            dry_run: If True, only return what would be deleted without actually deleting
        """
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.dry_run = dry_run

    def delete_file(self, file_path: Path) -> bool:
        """Delete file with retry. Returns True if deleted or doesn't exist.

        Args:
            file_path: Path to the file to delete

        Returns:
            True if file was deleted or doesn't exist, False on failure.
            In dry_run mode, returns True for files that exist (would be deleted).
        """
        # In dry_run mode, return True for existing files without deleting
        if self.dry_run:
            return file_path.exists()

        for attempt in range(self.max_retries):
            try:
                file_path.unlink()
                return True
            except PermissionError:
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_base * (attempt + 1))
                else:
                    logger.warning(f"File locked, could not delete: {file_path}")
                    return False
            except FileNotFoundError:
                return True  # Already deleted
            except OSError as e:
                logger.warning(f"OS error deleting {file_path}: {e}")
                return False
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")
                return False
        return False

    def delete_directory(self, dir_path: Path) -> bool:
        """Delete directory recursively with Windows permission handling.

        Handles read-only files by attempting to remove the readonly flag
        before deletion.

        Args:
            dir_path: Path to the directory to delete

        Returns:
            True if directory was deleted, False on failure.
            In dry_run mode, returns True for existing directories (would be deleted).
        """
        # In dry_run mode, return True for existing directories without deleting
        if self.dry_run:
            return dir_path.exists()

        def handle_remove_readonly(func, path, exc_info):
            """Error handler for shutil.rmtree - handles read-only files on Windows."""
            # Try to make the file writable and retry
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                pass  # Give up on this file, rmtree will continue

        try:
            shutil.rmtree(dir_path, onerror=handle_remove_readonly)
            return True
        except PermissionError as e:
            logger.warning(f"Permission denied removing directory {dir_path}: {e}")
            return False
        except OSError as e:
            logger.warning(f"OS error removing directory {dir_path}: {e}")
            return False
        except Exception as e:
            logger.warning(f"Failed to remove directory {dir_path}: {e}")
            return False

    def delete_files(self, file_paths: list[Path]) -> dict:
        """Delete multiple files at once.

        Args:
            file_paths: List of file paths to delete

        Returns:
            Dict with 'deleted' (list of successfully deleted paths),
            'failed' (list of paths that failed to delete),
            'not_found' (list of paths that didn't exist)
        """
        result = {
            'deleted': [],
            'failed': [],
            'not_found': []
        }

        for file_path in file_paths:
            if not file_path.exists():
                result['not_found'].append(file_path)
                continue

            if self.delete_file(file_path):
                result['deleted'].append(file_path)
            else:
                result['failed'].append(file_path)

        return result

    def delete_directories(self, dir_paths: list[Path]) -> dict:
        """Delete multiple directories at once.

        Args:
            dir_paths: List of directory paths to delete

        Returns:
            Dict with 'deleted' (list of successfully deleted paths),
            'failed' (list of paths that failed to delete)
        """
        result = {
            'deleted': [],
            'failed': []
        }

        for dir_path in dir_paths:
            if self.delete_directory(dir_path):
                result['deleted'].append(dir_path)
            else:
                result['failed'].append(dir_path)

        return result
