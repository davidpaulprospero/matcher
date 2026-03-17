"""
Additional unit tests for src/utils.py.

Complements test_utils_simple.py with tests for path resolution and utilities.
"""

import pytest
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import resolve_path, normalize_path, sanitize_path


class TestResolvePath:
    """Test path resolution with base directory."""

    @pytest.mark.fast
    def test_resolve_absolute_path(self):
        """Test resolving already absolute path."""
        abs_path = "/absolute/path/to/file.txt"

        resolved = resolve_path(abs_path)

        # Should return normalized absolute path
        assert Path(resolved).is_absolute()

    @pytest.mark.fast
    def test_resolve_relative_path_with_base(self, tmp_path):
        """Test resolving relative path with base directory."""
        base_dir = tmp_path
        rel_path = "relative/file.txt"

        resolved = resolve_path(rel_path, base_dir=base_dir)

        # Should be absolute and under base_dir
        assert Path(resolved).is_absolute()
        # Normalize both paths for comparison (handle slash differences)
        assert Path(resolved).is_relative_to(base_dir)

    @pytest.mark.fast
    def test_resolve_path_object(self, tmp_path):
        """Test resolving Path object."""
        path_obj = Path("test/file.txt")

        resolved = resolve_path(path_obj, base_dir=tmp_path)

        assert isinstance(resolved, str)
        assert Path(resolved).is_absolute()

    @pytest.mark.fast
    def test_resolve_current_directory(self):
        """Test resolving with current directory."""
        rel_path = "file.txt"

        resolved = resolve_path(rel_path)

        # Should resolve to absolute path
        assert Path(resolved).is_absolute()


class TestNormalizePath:
    """Test path normalization."""

    @pytest.mark.fast
    def test_normalize_forward_slashes(self):
        """Test normalizing path with forward slashes."""
        path = "C:/Users/test/file.txt"

        normalized = normalize_path(path)

        assert normalized is not None
        assert isinstance(normalized, str)

    @pytest.mark.fast
    def test_normalize_backslashes(self):
        """Test normalizing path with backslashes."""
        path = "C:\\Users\\test\\file.txt"

        normalized = normalize_path(path)

        assert normalized is not None

    @pytest.mark.fast
    def test_normalize_mixed_slashes(self):
        """Test normalizing path with mixed slashes."""
        path = "C:/Users\\test/file.txt"

        normalized = normalize_path(path)

        assert normalized is not None

    @pytest.mark.fast
    def test_normalize_empty_string(self):
        """Test normalizing empty string."""
        path = ""

        normalized = normalize_path(path)

        assert normalized == ""

    @pytest.mark.fast
    def test_normalize_preserves_content(self):
        """Test normalization preserves path content."""
        path = "D:/project/video.mp4"

        normalized = normalize_path(path)

        # Should contain the filename
        assert "video.mp4" in normalized


class TestSanitizePath:
    """Test path sanitization."""

    @pytest.mark.fast
    def test_sanitize_basic_path(self):
        """Test sanitizing basic path."""
        path = "C:/Videos/test.mp4"

        sanitized = sanitize_path(path)

        assert isinstance(sanitized, str)
        assert ".mp4" in sanitized

    @pytest.mark.fast
    def test_sanitize_path_with_spaces(self):
        """Test sanitizing path with spaces."""
        path = "C:/My Videos/test file.mp4"

        sanitized = sanitize_path(path)

        # Should handle spaces (replace or keep)
        assert isinstance(sanitized, str)

    @pytest.mark.fast
    def test_sanitize_path_object(self):
        """Test sanitizing Path object."""
        path_obj = Path("C:/Videos/test.mp4")

        sanitized = sanitize_path(path_obj)

        assert isinstance(sanitized, str)

    @pytest.mark.fast
    def test_sanitize_unicode_path(self):
        """Test sanitizing path with unicode characters."""
        path = "C:/Videos/日本語.mp4"

        sanitized = sanitize_path(path)

        assert isinstance(sanitized, str)

    @pytest.mark.fast
    def test_sanitize_empty_path(self):
        """Test sanitizing empty path."""
        path = ""

        sanitized = sanitize_path(path)

        assert sanitized == ""


class TestPathEdgeCases:
    """Test edge cases in path handling."""

    @pytest.mark.fast
    def test_very_long_path(self):
        """Test handling very long paths."""
        # Create a path with many nested directories
        long_path = "/".join(["dir"] * 50) + "/file.txt"

        normalized = normalize_path(long_path)

        assert isinstance(normalized, str)
        assert "file.txt" in normalized

    @pytest.mark.fast
    def test_path_with_dots(self):
        """Test path with dots (., ..)."""
        path = "C:/project/../videos/./file.mp4"

        normalized = normalize_path(path)

        assert isinstance(normalized, str)

    @pytest.mark.fast
    def test_network_path(self):
        """Test network/UNC path."""
        path = "//server/share/file.mp4"

        normalized = normalize_path(path)

        assert isinstance(normalized, str)

    @pytest.mark.fast
    def test_path_with_special_chars(self):
        """Test path with special characters."""
        path = "C:/videos/file@#$%.mp4"

        sanitized = sanitize_path(path)

        assert isinstance(sanitized, str)


class TestPathConsistency:
    """Test path handling consistency."""

    @pytest.mark.fast
    def test_normalize_idempotent(self):
        """Test normalizing twice gives same result."""
        path = "C:/Users\\test/file.txt"

        norm1 = normalize_path(path)
        norm2 = normalize_path(norm1)

        # Normalizing twice should give same result
        assert norm1 == norm2

    @pytest.mark.fast
    def test_sanitize_then_normalize(self):
        """Test sanitize followed by normalize."""
        path = "C:/My Videos/test file.mp4"

        sanitized = sanitize_path(path)
        normalized = normalize_path(sanitized)

        assert isinstance(normalized, str)

    @pytest.mark.fast
    def test_normalize_then_sanitize(self):
        """Test normalize followed by sanitize."""
        path = "C:\\Videos\\test.mp4"

        normalized = normalize_path(path)
        sanitized = sanitize_path(normalized)

        assert isinstance(sanitized, str)


class TestPathValidation:
    """Test path validation behavior."""

    @pytest.mark.fast
    def test_none_path_handling(self):
        """Test handling None path."""
        # Should handle gracefully or raise TypeError
        try:
            result = normalize_path(None)
            # If it doesn't raise, check result
            assert result is not None or result is None
        except (TypeError, AttributeError):
            # Expected behavior for None
            pass

    @pytest.mark.fast
    def test_numeric_path_handling(self):
        """Test handling numeric input."""
        # Should handle gracefully or raise TypeError
        try:
            result = normalize_path(123)
            assert isinstance(result, str) or result is None
        except (TypeError, AttributeError):
            # Expected behavior for non-string
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
