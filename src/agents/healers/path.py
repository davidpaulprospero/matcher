"""
Path Healer - File path length and encoding error recovery.

Handles:
- Windows 260 character path limit
- Unicode path issues
- Invalid characters in paths
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..base import Healer, HealerResult, HealerAction

if TYPE_CHECKING:
    from ...config import Config
    from ...state import PipelineState

logger = logging.getLogger(__name__)


class PathHealer(Healer):
    """
    Heals path-related errors.

    Recovery strategies:
    1. Path too long: Switch to short root paths (E:/v, E:/i)
    2. Unicode errors: Sanitize filenames
    3. Invalid characters: Replace problematic chars
    """

    name = "path-healer"
    description = "Fix path length and encoding errors"

    error_patterns = [
        "path too long",
        "filename too long",
        "name too long",
        "errno 63",   # ENAMETOOLONG
        "errno 36",   # ENAMETOOLONG (macOS)
        "errno 206",  # Windows path too long
        "unicode",
        "encode",
        "decode",
        "invalid path",
        "illegal character",
    ]

    # Windows MAX_PATH limit
    MAX_PATH_LENGTH = 260

    # Short path roots to try
    SHORT_ROOTS = ["E:/v", "D:/v", "C:/v", "E:/i", "D:/i", "C:/i"]

    def fix(
        self,
        error: Exception,
        state: 'PipelineState',
        stage_name: str
    ) -> HealerResult:
        """Attempt to fix path-related errors."""
        error_str = str(error).lower()

        # Path too long
        if any(p in error_str for p in ["path too long", "name too long", "errno 63", "errno 36", "errno 206"]):
            return self._handle_path_too_long(error, state)

        # Unicode/encoding errors
        if any(p in error_str for p in ["unicode", "encode", "decode"]):
            return self._handle_unicode_error(error, state)

        # Invalid characters
        if any(p in error_str for p in ["invalid path", "illegal character"]):
            return self._handle_invalid_chars(error, state)

        # Generic path error
        return self._handle_path_too_long(error, state)

    def _handle_path_too_long(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle path too long errors by switching to short roots."""
        self.log_attempt("Path too long, attempting to use short path roots...")

        # Find an available short root
        short_root = self._find_available_short_root()

        if not short_root:
            # Try to create one
            for root in self.SHORT_ROOTS:
                try:
                    Path(root).mkdir(parents=True, exist_ok=True)
                    short_root = root
                    break
                except Exception:
                    continue

        if short_root:
            # Update config to use short paths
            download_config = getattr(self.config, 'download', None)
            if download_config:
                if hasattr(download_config, 'root_dir'):
                    old_root = download_config.root_dir
                    download_config.root_dir = short_root
                elif isinstance(download_config, dict):
                    old_root = download_config.get('root_dir', 'default')
                    download_config['root_dir'] = short_root
                else:
                    old_root = 'default'

                self.log_success(f"Switched to short path root: {short_root}")
                return HealerResult.config_changed(
                    f"Switched download.root_dir to {short_root}",
                    old_root=old_root,
                    new_root=short_root
                )

        # No short roots available - try truncating filenames
        return self._truncate_filenames(error, state)

    def _handle_unicode_error(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle unicode encoding errors in paths."""
        self.log_attempt("Unicode error in path, sanitizing...")

        fixed_count = 0

        # Sanitize video/audio download paths
        if hasattr(state, 'downloads') and state.downloads:
            for download in state.downloads:
                if hasattr(download, 'file'):
                    original = download.file
                    sanitized = self._sanitize_unicode(original)
                    if sanitized != original:
                        download.file = sanitized
                        fixed_count += 1

        # Sanitize caption download paths (caption-first mode)
        if hasattr(state, 'caption_downloads') and state.caption_downloads:
            for caption in state.caption_downloads:
                if hasattr(caption, 'file'):
                    original = caption.file
                    sanitized = self._sanitize_unicode(original)
                    if sanitized != original:
                        caption.file = sanitized
                        fixed_count += 1

        # Sanitize audio download paths
        if hasattr(state, 'downloaded_audio') and state.downloaded_audio:
            for audio in state.downloaded_audio:
                if hasattr(audio, 'file'):
                    original = audio.file
                    sanitized = self._sanitize_unicode(original)
                    if sanitized != original:
                        audio.file = sanitized
                        fixed_count += 1

        if fixed_count > 0:
            self.log_success(f"Sanitized {fixed_count} file paths")
            return HealerResult.fixed(
                f"Sanitized {fixed_count} paths with unicode issues",
                action=HealerAction.RETRY,
                fixed_count=fixed_count
            )

        return HealerResult.failed("Could not find paths to sanitize")

    def _handle_invalid_chars(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle invalid characters in paths."""
        self.log_attempt("Invalid characters in path, cleaning...")

        return self._handle_unicode_error(error, state)

    def _truncate_filenames(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Truncate long filenames as a last resort."""
        self.log_attempt("Truncating long filenames...")

        # Update config to use shorter filenames
        download_config = getattr(self.config, 'download', None)
        if download_config:
            # Set a max filename length
            max_len = 50  # Conservative limit

            if hasattr(download_config, 'max_filename_length'):
                download_config.max_filename_length = max_len
            elif isinstance(download_config, dict):
                download_config['max_filename_length'] = max_len

            self.log_success(f"Set max filename length to {max_len}")
            return HealerResult.config_changed(
                f"Limiting filenames to {max_len} characters",
                max_filename_length=max_len
            )

        return HealerResult.failed("Could not configure filename truncation")

    def _find_available_short_root(self) -> Optional[str]:
        """Find an existing short path root that's writable."""
        for root in self.SHORT_ROOTS:
            path = Path(root)
            if path.exists() and path.is_dir():
                # Check if writable
                test_file = path / ".write_test"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                    return root
                except Exception:
                    continue
        return None

    def _sanitize_unicode(self, path_str: str) -> str:
        """Remove or replace problematic unicode characters."""
        if not path_str:
            return path_str

        # Common unicode replacements
        replacements = {
            '\u2019': "'",   # Right single quote
            '\u2018': "'",   # Left single quote
            '\u201c': '"',   # Left double quote
            '\u201d': '"',   # Right double quote
            '\u2013': '-',   # En dash
            '\u2014': '-',   # Em dash
            '\u2026': '...',  # Ellipsis
            '\u00a0': ' ',   # Non-breaking space
        }

        result = path_str
        for char, replacement in replacements.items():
            result = result.replace(char, replacement)

        # Remove remaining non-ASCII from filename only (preserve path separators)
        path = Path(result)
        clean_name = re.sub(r'[^\x00-\x7F]+', '_', path.name)

        # Also remove Windows-invalid characters
        invalid_chars = '<>:"|?*'
        for char in invalid_chars:
            clean_name = clean_name.replace(char, '_')

        return str(path.parent / clean_name)

    def check_path_length(self, path: str) -> bool:
        """Check if a path exceeds Windows MAX_PATH."""
        return len(path) <= self.MAX_PATH_LENGTH

    def estimate_safe_filename_length(self, directory: Path) -> int:
        """Estimate maximum safe filename length given a directory."""
        dir_len = len(str(directory))
        # Leave room for path separator and some buffer
        available = self.MAX_PATH_LENGTH - dir_len - 10
        return max(20, min(available, 200))  # Between 20 and 200 chars
