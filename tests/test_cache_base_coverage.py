"""
Tests for src/cache/base.py edge cases and error handling.

Targets uncovered lines:
- Lines 110-112: JSONDecodeError/IOError when loading cache index
- Lines 126-127: IOError when saving cache index
- Line 179: Entry without cached_at attribute
- Lines 206-208: Exception when deserializing cache entry
- Lines 296-298: Exception when validating entry in cleanup
- Lines 329-330: OSError when getting file stats
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from dataclasses import dataclass
from typing import Any, Dict

# Import the cache base module
from src.cache.base import BaseCache, CacheEntry


# Create a concrete implementation for testing
class ConcreteCache(BaseCache[str]):
    """Concrete cache implementation for testing."""

    def _get_default_index(self) -> Dict[str, Any]:
        return {"entries": {}}

    def _count_entries(self) -> int:
        return len(self.index.get("entries", {}))

    def _serialize_entry(self, entry: CacheEntry[str]) -> Dict[str, Any]:
        return {
            "value": entry.value,
            "cached_at": entry.cached_at,
            "metadata": entry.metadata
        }

    def _deserialize_entry(self, data: Dict[str, Any]) -> CacheEntry[str]:
        if "corrupt" in str(data.get("value", "")):
            raise ValueError("Corrupt entry!")
        return CacheEntry(
            value=data.get("value", ""),
            cached_at=data.get("cached_at", ""),
            metadata=data.get("metadata", {})
        )


class TestCacheIndexLoadingErrors:
    """Test cache index loading error handling (lines 110-112)."""

    @pytest.mark.fast
    def test_json_decode_error_starts_fresh_lines_110_112(self, tmp_path):
        """Test lines 110-112: JSONDecodeError when loading index starts fresh."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Default index_name is "index.json"
        index_path = cache_dir / "index.json"

        # Write corrupt JSON
        index_path.write_text("{ invalid json without closing brace")

        # Create cache - should handle error and start fresh
        cache = ConcreteCache(cache_dir=cache_dir)

        # Should have default/fresh index
        assert cache.index == {"entries": {}}

    @pytest.mark.fast
    def test_io_error_starts_fresh_line_110(self, tmp_path):
        """Test line 110: IOError when loading index starts fresh."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        index_path = cache_dir / "cache_index.json"

        # Write valid JSON so the file exists
        index_path.write_text('{"entries": {"key1": "value1"}}')

        # Mock open to raise IOError during read
        with patch('builtins.open', side_effect=IOError("Disk read error")):
            cache = ConcreteCache(cache_dir=cache_dir)

        # Should have default/fresh index due to IOError
        assert cache.index == {"entries": {}}


class TestCacheIndexSaveErrors:
    """Test cache index save error handling (lines 126-127)."""

    @pytest.mark.fast
    def test_io_error_on_save_lines_126_127(self, tmp_path):
        """Test lines 126-127: IOError when saving cache index is logged."""
        cache_dir = tmp_path / "cache"
        cache = ConcreteCache(cache_dir=cache_dir)

        # Mock open to raise IOError during write
        with patch.object(Path, 'mkdir'):
            with patch('builtins.open', side_effect=IOError("Disk full")):
                # Should not raise, just log error
                cache._save_index()


class TestCacheEntryValidation:
    """Test cache entry validation (line 179)."""

    @pytest.mark.fast
    def test_entry_without_cached_at_is_valid_line_179(self, tmp_path):
        """Test line 179: Entry without cached_at attribute returns True."""
        cache_dir = tmp_path / "cache"
        cache = ConcreteCache(cache_dir=cache_dir)

        # Create entry-like object without cached_at
        @dataclass
        class EntryWithoutCachedAt:
            value: str
            metadata: dict

        entry = EntryWithoutCachedAt(value="test", metadata={})

        # Should return True (line 179)
        assert cache._is_valid_entry(entry) is True


class TestCacheGetDeserializationError:
    """Test cache get deserialization error (lines 206-208)."""

    @pytest.mark.fast
    def test_deserialization_exception_returns_none_lines_206_208(self, tmp_path):
        """Test lines 206-208: Exception during deserialization returns None."""
        cache_dir = tmp_path / "cache"
        cache = ConcreteCache(cache_dir=cache_dir)

        # Add a corrupt entry to index
        cache.index = {
            "entries": {},
            "corrupt_key": {"value": "corrupt data trigger"}
        }

        # get should catch exception and return None
        result = cache.get("corrupt_key")
        assert result is None


class TestCacheCleanupErrors:
    """Test cache cleanup error handling (lines 296-298)."""

    @pytest.mark.fast
    def test_cleanup_validation_exception_lines_296_298(self, tmp_path):
        """Test lines 296-298: Exception during entry validation in cleanup."""
        cache_dir = tmp_path / "cache"
        cache = ConcreteCache(cache_dir=cache_dir)

        # Add entries including one that will cause deserialization error
        cache.index = {
            "good_key": {"value": "valid", "cached_at": "2026-01-01T00:00:00"},
            "bad_key": {"value": "corrupt data trigger", "cached_at": "2026-01-01T00:00:00"}
        }

        # cleanup_expired should catch exception and mark bad_key for removal
        removed = cache.cleanup_expired()

        # bad_key should have been removed
        assert "bad_key" not in cache.index
        assert removed >= 1


class TestCacheGetStatsOSError:
    """Test cache get_stats OSError handling (lines 329-330)."""

    @pytest.mark.fast
    def test_file_stat_os_error_lines_329_330(self, tmp_path):
        """Test lines 329-330: OSError when getting file stats is ignored."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create some files
        (cache_dir / "file1.txt").write_text("data1")
        (cache_dir / "file2.txt").write_text("data2")

        cache = ConcreteCache(cache_dir=cache_dir)

        # Keep track of call count per file to allow is_file() to work
        # but fail on the second stat call (inside the try block)
        call_counts = {}
        original_stat = Path.stat

        def mock_stat(self):
            if self.name == "file2.txt":
                call_counts.setdefault(self.name, 0)
                call_counts[self.name] += 1
                # First call is from is_file(), let it through
                # Second call is from cache_size calculation, fail it
                if call_counts[self.name] > 1:
                    raise OSError("Permission denied")
            return original_stat(self)

        with patch.object(Path, 'stat', mock_stat):
            stats = cache.get_stats()

        # Should still return stats (just with partial size info)
        assert stats['total_entries'] >= 0
        assert 'cache_size_mb' in stats
