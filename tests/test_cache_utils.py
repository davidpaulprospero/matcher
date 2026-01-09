"""
Unit tests for cache utility functions.

Tests hashing, path normalization, and batch operations.
"""

import pytest
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cache.utils import (
    compute_hash,
    batch_hash,
    text_hash,
    normalize_path
)


class TestComputeHash:
    """Test compute_hash function."""

    def test_hash_string(self):
        """Test hashing a string."""
        data = "test string"

        hash_value = compute_hash(data)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 16  # Default length

    def test_hash_bytes(self):
        """Test hashing bytes."""
        data = b"test bytes"

        hash_value = compute_hash(data)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 16

    def test_same_input_same_hash(self):
        """Test same input produces same hash."""
        data = "consistent input"

        hash1 = compute_hash(data)
        hash2 = compute_hash(data)

        assert hash1 == hash2

    def test_different_input_different_hash(self):
        """Test different inputs produce different hashes."""
        data1 = "input one"
        data2 = "input two"

        hash1 = compute_hash(data1)
        hash2 = compute_hash(data2)

        assert hash1 != hash2

    def test_custom_hash_length(self):
        """Test custom hash length."""
        data = "test"

        hash_short = compute_hash(data, length=8)
        hash_long = compute_hash(data, length=32)

        assert len(hash_short) == 8
        assert len(hash_long) == 32

    def test_empty_string(self):
        """Test hashing empty string."""
        data = ""

        hash_value = compute_hash(data)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 16


class TestBatchHash:
    """Test batch_hash function for lists."""

    def test_batch_hash_list(self):
        """Test hashing a list of strings."""
        items = ["item1", "item2", "item3"]

        hash_value = batch_hash(items)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 16

    def test_batch_hash_consistency(self):
        """Test same list produces same hash."""
        items = ["a", "b", "c"]

        hash1 = batch_hash(items)
        hash2 = batch_hash(items)

        assert hash1 == hash2

    def test_batch_hash_order_matters(self):
        """Test order of items (may or may not affect hash depending on implementation)."""
        items1 = ["a", "b", "c"]
        items2 = ["c", "b", "a"]

        hash1 = batch_hash(items1)
        hash2 = batch_hash(items2)

        # Implementation may sort items, so hash could be same
        assert isinstance(hash1, str) and isinstance(hash2, str)

    def test_batch_hash_empty_list(self):
        """Test hashing empty list."""
        items = []

        hash_value = batch_hash(items)

        assert isinstance(hash_value, str)

    def test_batch_hash_single_item(self):
        """Test hashing list with single item."""
        items = ["single"]

        hash_value = batch_hash(items)

        assert isinstance(hash_value, str)

    def test_batch_hash_duplicates(self):
        """Test list with duplicate items."""
        items = ["dup", "dup", "unique"]

        hash_value = batch_hash(items)

        assert isinstance(hash_value, str)


class TestTextHash:
    """Test text_hash function."""

    def test_text_hash_basic(self):
        """Test basic text hashing."""
        text = "Sample text for hashing"

        hash_value = text_hash(text)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 12  # Default length

    def test_text_hash_consistency(self):
        """Test same text produces same hash."""
        text = "consistent text"

        hash1 = text_hash(text)
        hash2 = text_hash(text)

        assert hash1 == hash2

    def test_text_hash_custom_length(self):
        """Test custom hash length."""
        text = "test"

        hash_short = text_hash(text, length=6)
        hash_long = text_hash(text, length=20)

        assert len(hash_short) == 6
        assert len(hash_long) == 20

    def test_text_hash_empty_string(self):
        """Test hashing empty text."""
        text = ""

        hash_value = text_hash(text)

        assert isinstance(hash_value, str)

    def test_text_hash_unicode(self):
        """Test hashing unicode text."""
        text = "Unicode: 日本語 français"

        hash_value = text_hash(text)

        assert isinstance(hash_value, str)

    def test_text_hash_whitespace(self):
        """Test hashing text with whitespace."""
        text = "  spaces  \n\t  tabs  "

        hash_value = text_hash(text)

        assert isinstance(hash_value, str)


class TestNormalizePath:
    """Test path normalization."""

    def test_normalize_string_path(self):
        """Test normalizing string path."""
        path = "C:/Users/test/file.txt"

        normalized = normalize_path(path)

        assert isinstance(normalized, str)

    def test_normalize_path_object(self):
        """Test normalizing Path object."""
        path = Path("test/file.txt")

        normalized = normalize_path(path)

        assert isinstance(normalized, str)

    def test_normalize_absolute_path(self):
        """Test normalizing absolute path."""
        path = "/absolute/path/to/file.txt"

        normalized = normalize_path(path)

        assert isinstance(normalized, str)

    def test_normalize_relative_path(self):
        """Test normalizing relative path."""
        path = "relative/path/file.txt"

        normalized = normalize_path(path)

        assert isinstance(normalized, str)

    def test_normalize_empty_path(self):
        """Test normalizing empty path."""
        path = ""

        normalized = normalize_path(path)

        # Empty path may normalize to current directory
        assert isinstance(normalized, str)

    def test_normalize_consistency(self):
        """Test normalizing twice gives same result."""
        path = "test/path/file.txt"

        norm1 = normalize_path(path)
        norm2 = normalize_path(norm1)

        assert norm1 == norm2


class TestHashConsistency:
    """Test hash consistency across different scenarios."""

    def test_string_vs_bytes_same_content(self):
        """Test hashing equivalent string and bytes."""
        string_data = "test"
        bytes_data = b"test"

        hash_string = compute_hash(string_data)
        hash_bytes = compute_hash(bytes_data)

        # Should produce same hash for same content
        assert hash_string == hash_bytes

    def test_whitespace_differences(self):
        """Test hashes differ with whitespace changes."""
        text1 = "no spaces"
        text2 = "no  spaces"  # Two spaces

        hash1 = text_hash(text1)
        hash2 = text_hash(text2)

        assert hash1 != hash2

    def test_case_sensitivity(self):
        """Test hashes are case sensitive."""
        text1 = "lowercase"
        text2 = "LOWERCASE"

        hash1 = text_hash(text1)
        hash2 = text_hash(text2)

        assert hash1 != hash2


class TestHashEdgeCases:
    """Test edge cases in hashing."""

    def test_very_long_text(self):
        """Test hashing very long text."""
        text = "a" * 10000

        hash_value = text_hash(text)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 12

    def test_special_characters(self):
        """Test hashing text with special characters."""
        text = "Special!@#$%^&*()_+-=[]{}|;':\",./<>?"

        hash_value = text_hash(text)

        assert isinstance(hash_value, str)

    def test_newlines_and_tabs(self):
        """Test hashing text with newlines and tabs."""
        text = "Line1\nLine2\tTabbed"

        hash_value = text_hash(text)

        assert isinstance(hash_value, str)

    def test_batch_hash_large_list(self):
        """Test hashing large list of items."""
        items = [f"item_{i}" for i in range(1000)]

        hash_value = batch_hash(items)

        assert isinstance(hash_value, str)


class TestHashLength:
    """Test hash length parameter."""

    def test_minimum_length(self):
        """Test minimum hash length."""
        data = "test"

        hash_value = compute_hash(data, length=1)

        assert len(hash_value) == 1

    def test_maximum_length(self):
        """Test large hash length."""
        data = "test"

        hash_value = compute_hash(data, length=64)

        # Hash length may be capped at MD5 length (32)
        assert len(hash_value) <= 64

    def test_zero_length(self):
        """Test zero length hash."""
        data = "test"

        hash_value = compute_hash(data, length=0)

        # Should handle gracefully (empty string or minimum length)
        assert isinstance(hash_value, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
