"""
Disk Healer - Disk space and storage error recovery.

Handles:
- Disk full errors
- Write permission errors
- Cache cleanup
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple

from ..base import Healer, HealerResult, HealerAction

if TYPE_CHECKING:
    from ...config import Config
    from ...state import PipelineState

logger = logging.getLogger(__name__)


class DiskHealer(Healer):
    """
    Heals disk space and storage errors.

    Recovery strategies:
    1. Disk full: Clean caches, remove temp files
    2. Permission error: Suggest alternate location
    3. Low space warning: Proactive cleanup
    """

    name = "disk-healer"
    description = "Fix disk space and storage errors"

    error_patterns = [
        "disk full",
        "no space",
        "not enough space",
        "storage",
        "permission denied",
        "access denied",
        "read-only",
        "oserror",
        "errno 28",  # ENOSPC
        "errno 13",  # EACCES
    ]

    # Minimum free space to continue (1 GB)
    MIN_FREE_SPACE_GB = 1.0

    # Cache directories to clean (relative to project)
    CACHE_DIRS = [
        ".cache/llm_responses",
        ".cache/vision_cache",
        ".cache/embeddings",
    ]

    def fix(
        self,
        error: Exception,
        state: 'PipelineState',
        stage_name: str
    ) -> HealerResult:
        """Attempt to fix disk-related errors."""
        error_str = str(error).lower()

        # Disk full
        if any(p in error_str for p in ["disk full", "no space", "errno 28", "not enough"]):
            return self._handle_disk_full(error, state)

        # Permission errors
        if any(p in error_str for p in ["permission", "access denied", "errno 13", "read-only"]):
            return self._handle_permission_error(error, state)

        # Generic storage error
        return self._handle_disk_full(error, state)

    def _handle_disk_full(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle disk full errors by cleaning caches."""
        self.log_attempt("Disk full, attempting cleanup...")

        freed_bytes = 0
        cleaned_dirs = []

        # Clean project caches
        for cache_rel in self.CACHE_DIRS:
            cache_path = Path(self.project_dir) / cache_rel
            if cache_path.exists():
                size = self._get_dir_size(cache_path)
                if size > 0:
                    try:
                        shutil.rmtree(cache_path)
                        cache_path.mkdir(parents=True, exist_ok=True)
                        freed_bytes += size
                        cleaned_dirs.append(cache_rel)
                        self.log_attempt(f"Cleaned {cache_rel}: {self._format_size(size)}")
                    except Exception as e:
                        self.log_failure(f"Could not clean {cache_rel}: {e}")

        # Check global cache
        global_cache = Path.home() / ".matcher_global_cache"
        if global_cache.exists():
            old_files = self._find_old_files(global_cache, days=30)
            for f in old_files[:50]:  # Limit to 50 files at a time
                try:
                    size = f.stat().st_size
                    f.unlink()
                    freed_bytes += size
                except Exception:
                    pass
            if old_files:
                cleaned_dirs.append("global_cache (old files)")

        if freed_bytes > 0:
            freed_str = self._format_size(freed_bytes)
            self.log_success(f"Freed {freed_str} from: {', '.join(cleaned_dirs)}")
            return HealerResult.fixed(
                f"Freed {freed_str} by cleaning caches",
                action=HealerAction.RETRY,
                freed_bytes=freed_bytes,
                cleaned_dirs=cleaned_dirs
            )

        # Check if there's now enough space
        free_space = self._get_free_space(Path(self.project_dir))
        if free_space >= self.MIN_FREE_SPACE_GB * 1024 * 1024 * 1024:
            return HealerResult.fixed(
                f"Sufficient disk space available ({self._format_size(free_space)})",
                action=HealerAction.RETRY
            )

        self.log_failure("Could not free enough disk space")
        return HealerResult.failed(
            f"Disk full. Need at least {self.MIN_FREE_SPACE_GB} GB free. "
            f"Currently have {self._format_size(free_space)}."
        )

    def _handle_permission_error(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle permission errors."""
        self.log_attempt("Permission error detected...")

        # Check if we can write to project dir
        test_file = Path(self.project_dir) / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
            # Project dir is writable, error was elsewhere
            return HealerResult.fixed(
                "Project directory is writable, retrying operation",
                action=HealerAction.RETRY
            )
        except Exception:
            pass

        # Suggest using short paths
        self.log_failure("Cannot write to project directory")
        return HealerResult.failed(
            "Permission denied. Check that you have write access to the project directory, "
            "or try using short paths (E:/v, E:/i) in config.yaml"
        )

    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        try:
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        except Exception:
            pass
        return total

    def _get_free_space(self, path: Path) -> int:
        """Get free space on the drive containing path."""
        try:
            usage = shutil.disk_usage(path)
            return usage.free
        except Exception:
            return 0

    def _find_old_files(self, directory: Path, days: int = 30) -> List[Path]:
        """Find files older than specified days."""
        import time

        cutoff = time.time() - (days * 24 * 60 * 60)
        old_files = []

        try:
            for f in directory.rglob("*"):
                if f.is_file():
                    try:
                        if f.stat().st_mtime < cutoff:
                            old_files.append(f)
                    except Exception:
                        pass
        except Exception:
            pass

        # Sort by size (largest first)
        old_files.sort(key=lambda f: f.stat().st_size, reverse=True)
        return old_files

    def _format_size(self, bytes: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes < 1024:
                return f"{bytes:.1f} {unit}"
            bytes /= 1024
        return f"{bytes:.1f} TB"

    def check_space(self, required_gb: float = None) -> Tuple[bool, float]:
        """
        Check if there's enough disk space.

        Returns:
            Tuple of (has_enough_space, free_space_gb)
        """
        required = required_gb or self.MIN_FREE_SPACE_GB
        free_bytes = self._get_free_space(Path(self.project_dir))
        free_gb = free_bytes / (1024 * 1024 * 1024)
        return (free_gb >= required, free_gb)
