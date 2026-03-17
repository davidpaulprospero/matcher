"""
Tests for src/utils.py path and embeddings utility functions.

Tests utility functions for:
- is_embeddings_empty() - safe embeddings checking
- sanitize_path() - path cleaning and normalization
- normalize_path() - cross-platform path matching
- resolve_path() - absolute path resolution
"""

import pytest

# This module uses os.chdir() and must run serially to avoid affecting other tests
pytestmark = pytest.mark.serial
import sys
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    is_embeddings_empty,
    sanitize_path,
    normalize_path,
    resolve_path
)


class TestIsEmbeddingsEmpty:
    """Test is_embeddings_empty() function"""

    @pytest.mark.fast
    def test_is_embeddings_empty_none(self):
        """Test with None input"""
        assert is_embeddings_empty(None) is True

    @pytest.mark.fast
    def test_is_embeddings_empty_empty_list(self):
        """Test with empty list"""
        assert is_embeddings_empty([]) is True

    @pytest.mark.fast
    def test_is_embeddings_empty_empty_numpy_array(self):
        """Test with empty numpy array"""
        embeddings = np.array([])
        assert is_embeddings_empty(embeddings) is True

    @pytest.mark.fast
    def test_is_embeddings_empty_non_empty_list(self):
        """Test with non-empty list"""
        embeddings = [1.0, 2.0, 3.0]
        assert is_embeddings_empty(embeddings) is False

    @pytest.mark.fast
    def test_is_embeddings_empty_non_empty_numpy_array(self):
        """Test with non-empty numpy array"""
        embeddings = np.array([1.0, 2.0, 3.0])
        assert is_embeddings_empty(embeddings) is False

    @pytest.mark.fast
    def test_is_embeddings_empty_multidimensional_array(self):
        """Test with multidimensional numpy array"""
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert is_embeddings_empty(embeddings) is False

    @pytest.mark.fast
    def test_is_embeddings_empty_empty_multidimensional_array(self):
        """Test with empty multidimensional array"""
        embeddings = np.array([[]])
        # Has length 1 (one row), not empty
        assert is_embeddings_empty(embeddings) is False

    @pytest.mark.fast
    def test_is_embeddings_empty_zero_shape_array(self):
        """Test with (0,) shape array"""
        embeddings = np.zeros((0,))
        assert is_embeddings_empty(embeddings) is True

    @pytest.mark.fast
    def test_is_embeddings_empty_tuple(self):
        """Test with tuple"""
        embeddings = (1.0, 2.0, 3.0)
        assert is_embeddings_empty(embeddings) is False

    @pytest.mark.fast
    def test_is_embeddings_empty_empty_tuple(self):
        """Test with empty tuple"""
        embeddings = ()
        assert is_embeddings_empty(embeddings) is True


class TestSanitizePath:
    """Test sanitize_path() function"""

    @pytest.mark.fast
    def test_sanitize_path_windows_extended_length_prefix(self):
        r"""Test removing \\?\ prefix"""
        path = r'\\?\C:\Users\test\file.txt'
        result = sanitize_path(path)

        assert result == 'C:/Users/test/file.txt'
        assert not result.startswith(r'\\?')

    @pytest.mark.fast
    def test_sanitize_path_device_form_prefix(self):
        r"""Test removing \\.\ prefix"""
        path = r'\\.\C:\path\to\file.txt'
        result = sanitize_path(path)

        assert 'C:/path/to/file.txt' in result
        assert not result.startswith(r'\\.')

    @pytest.mark.fast
    def test_sanitize_path_forward_slash_extended(self):
        """Test removing //? / prefix"""
        path = '//?/C:/path/file.txt'
        result = sanitize_path(path)

        assert 'C:/path/file.txt' in result

    @pytest.mark.fast
    def test_sanitize_path_question_mark_prefix(self):
        r"""Test removing ?\ prefix"""
        path = r'?\C:\path\file.txt'
        result = sanitize_path(path)

        assert 'C:/path/file.txt' in result
        assert not result.startswith('?')

    @pytest.mark.fast
    def test_sanitize_path_backslash_to_forward_slash(self):
        """Test converting backslashes to forward slashes"""
        path = r'C:\Users\test\Documents\file.txt'
        result = sanitize_path(path)

        assert result == 'C:/Users/test/Documents/file.txt'
        assert '\\' not in result

    @pytest.mark.fast
    def test_sanitize_path_remove_double_slashes(self):
        """Test removing double slashes"""
        path = 'C://Users//test//file.txt'
        result = sanitize_path(path)

        assert '//' not in result
        assert result == 'C:/Users/test/file.txt'

    @pytest.mark.fast
    def test_sanitize_path_unix_path(self):
        """Test Unix-style path"""
        path = '/home/user/documents/file.txt'
        result = sanitize_path(path)

        assert result == '/home/user/documents/file.txt'

    @pytest.mark.fast
    def test_sanitize_path_with_path_object(self):
        """Test with Path object input"""
        path = Path(r'C:\Users\test\file.txt')
        result = sanitize_path(path)

        assert '/' in result
        assert '\\' not in result

    @pytest.mark.fast
    def test_sanitize_path_relative_path(self):
        """Test with relative path"""
        path = r'docs\readme.txt'
        result = sanitize_path(path)

        assert result == 'docs/readme.txt'

    @pytest.mark.fast
    def test_sanitize_path_network_path(self):
        """Test with network path"""
        path = r'\\server\share\file.txt'
        result = sanitize_path(path)

        # Should handle network paths
        assert isinstance(result, str)

    @pytest.mark.fast
    def test_sanitize_path_empty_string(self):
        """Test with empty string"""
        result = sanitize_path('')

        assert result == ''


class TestNormalizePath:
    """Test normalize_path() function"""

    @pytest.mark.fast
    def test_normalize_path_backslash_to_forward(self):
        """Test converting backslashes to forward slashes"""
        path = r'C:\Users\test\file.txt'
        result = normalize_path(path)

        assert result == 'c:/users/test/file.txt'
        assert '\\' not in result

    @pytest.mark.fast
    def test_normalize_path_lowercase(self):
        """Test converting to lowercase"""
        path = 'C:/Users/Test/FILE.TXT'
        result = normalize_path(path)

        assert result == 'c:/users/test/file.txt'
        assert result.islower()

    @pytest.mark.fast
    def test_normalize_path_unix_path(self):
        """Test Unix path normalization"""
        path = '/HOME/USER/Documents/File.txt'
        result = normalize_path(path)

        assert result == '/home/user/documents/file.txt'

    @pytest.mark.fast
    def test_normalize_path_mixed_slashes(self):
        """Test path with mixed slashes"""
        path = r'C:\Users/test\Documents/file.txt'
        result = normalize_path(path)

        assert result == 'c:/users/test/documents/file.txt'
        assert '\\' not in result

    @pytest.mark.fast
    def test_normalize_path_empty_string(self):
        """Test with empty string"""
        result = normalize_path('')

        assert result == ''

    @pytest.mark.fast
    def test_normalize_path_none_like(self):
        """Test with None-like values"""
        result = normalize_path(None)

        assert result == ''

    @pytest.mark.fast
    def test_normalize_path_relative(self):
        """Test with relative path"""
        path = r'docs\README.MD'
        result = normalize_path(path)

        assert result == 'docs/readme.md'

    @pytest.mark.fast
    def test_normalize_path_consistency(self):
        """Test that different representations normalize to same value"""
        path1 = r'C:\Users\test\file.txt'
        path2 = 'C:/Users/Test/file.txt'
        path3 = 'c:/users/TEST/FILE.TXT'

        result1 = normalize_path(path1)
        result2 = normalize_path(path2)
        result3 = normalize_path(path3)

        assert result1 == result2 == result3


class TestResolvePath:
    """Test resolve_path() function"""

    @pytest.mark.fast
    def test_resolve_path_absolute_path(self):
        """Test with absolute path"""
        path = 'C:/Users/test/file.txt'
        result = resolve_path(path)

        assert isinstance(result, str)
        assert '/' in result  # Sanitized

    @pytest.mark.fast
    def test_resolve_path_relative_with_base(self):
        """Test relative path with base directory"""
        path = 'subdir/file.txt'
        base_dir = 'C:/Users/test'

        result = resolve_path(path, base_dir)

        assert 'subdir/file.txt' in result
        assert isinstance(result, str)

    @pytest.mark.fast
    def test_resolve_path_path_object(self):
        """Test with Path object"""
        path = Path('file.txt')
        result = resolve_path(path)

        assert isinstance(result, str)
        assert '/' in result or result == 'file.txt'

    @pytest.mark.fast
    def test_resolve_path_current_directory(self, tmp_path):
        """Test resolving from current directory"""
        # Create test file
        test_file = tmp_path / 'test.txt'
        test_file.write_text('test')

        # Change to temp directory
        import os
        original_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = resolve_path('test.txt')

            # Should resolve to absolute path
            assert 'test.txt' in result
            assert isinstance(result, str)
        finally:
            os.chdir(original_dir)

    @pytest.mark.fast
    def test_resolve_path_sanitizes_output(self):
        """Test that output is sanitized"""
        path = r'C:\Users\test\file.txt'
        result = resolve_path(path)

        # Should have forward slashes
        assert '\\' not in result

    @pytest.mark.fast
    def test_resolve_path_with_base_path_object(self):
        """Test with Path object as base_dir"""
        path = 'file.txt'
        base_dir = Path('C:/Users/test')

        result = resolve_path(path, base_dir)

        assert isinstance(result, str)
        assert 'file.txt' in result

    @pytest.mark.fast
    def test_resolve_path_handles_symlinks(self, tmp_path):
        """Test resolving symlinks"""
        # Create a file and directory
        real_file = tmp_path / 'real.txt'
        real_file.write_text('content')

        result = resolve_path(str(real_file))

        assert isinstance(result, str)
        assert 'real.txt' in result


class TestPathEdgeCases:
    """Test edge cases for path utilities"""

    @pytest.mark.fast
    def test_sanitize_multiple_prefixes(self):
        """Test path with multiple prefix patterns"""
        # Edge case: path that might trigger multiple prefix checks
        path = r'\\?\?\C:\path\file.txt'
        result = sanitize_path(path)

        assert isinstance(result, str)
        assert '?' not in result[:5]  # No ? in first 5 chars

    @pytest.mark.fast
    def test_normalize_unicode_path(self):
        """Test normalization with Unicode characters"""
        path = 'C:/Users/日本語/文件.txt'
        result = normalize_path(path)

        assert isinstance(result, str)
        assert '日本語' in result

    @pytest.mark.fast
    def test_sanitize_very_long_path(self):
        """Test with very long path (>260 chars on Windows)"""
        # Create a path > 260 characters
        long_name = 'a' * 100
        path = f'C:/Users/test/{long_name}/{long_name}/{long_name}/file.txt'

        result = sanitize_path(path)

        assert isinstance(result, str)
        assert len(result) > 260

    @pytest.mark.fast
    def test_resolve_nonexistent_path_with_base(self):
        """Test resolving nonexistent path with base dir"""
        path = 'nonexistent/path/file.txt'
        base_dir = 'C:/Users/test'

        result = resolve_path(path, base_dir)

        # Should still return a path
        assert isinstance(result, str)
        assert 'nonexistent' in result

    @pytest.mark.fast
    def test_embeddings_with_complex_nested_structure(self):
        """Test is_embeddings_empty with complex nested arrays"""
        embeddings = np.array([[[1.0, 2.0], [3.0, 4.0]]])

        result = is_embeddings_empty(embeddings)

        assert result is False

    @pytest.mark.fast
    def test_embeddings_with_single_element(self):
        """Test with single element array"""
        embeddings = np.array([1.0])

        result = is_embeddings_empty(embeddings)

        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
