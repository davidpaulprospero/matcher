"""
Unit tests for unified cache package.

Tests BaseCache functionality and migrated cache implementations.
Created: January 7, 2026
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import time

from src.cache import BaseCache, CacheEntry, compute_hash, normalize_path, batch_hash


class SimpleCacheTestImpl(BaseCache):
    """Test implementation of BaseCache"""

    def _serialize_entry(self, entry: CacheEntry) -> dict:
        return {
            'data': entry.data,
            'cached_at': entry.cached_at,
            'key': entry.key,
            'metadata': entry.metadata
        }

    def _deserialize_entry(self, data: dict) -> CacheEntry:
        return CacheEntry(
            data=data['data'],
            cached_at=data['cached_at'],
            key=data.get('key', ''),
            metadata=data.get('metadata', {})
        )


class TestBaseCache(unittest.TestCase):
    """Test BaseCache functionality"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.cache = SimpleCacheTestImpl(
            cache_dir=Path(self.temp_dir),
            ttl_seconds=2  # 2 second TTL for testing
        )

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_basic_get_set(self):
        """Test basic cache operations"""
        self.cache.set('key1', {'value': 'test'})
        result = self.cache.get('key1')

        self.assertIsNotNone(result)
        self.assertEqual(result.data, {'value': 'test'})

    def test_get_nonexistent(self):
        """Test getting non-existent key"""
        result = self.cache.get('nonexistent')
        self.assertIsNone(result)

    def test_ttl_expiration(self):
        """Test TTL expiration"""
        self.cache.set('key1', 'value1')

        # Should exist immediately
        self.assertIsNotNone(self.cache.get('key1'))

        # Wait for expiration
        time.sleep(2.5)

        # Should be expired
        self.assertIsNone(self.cache.get('key1'))

    def test_persistence(self):
        """Test cache persistence across instances"""
        self.cache.set('key1', 'value1')
        self.cache.set('key2', 'value2')

        # Create new cache instance
        cache2 = SimpleCacheTestImpl(cache_dir=Path(self.temp_dir))

        # Should load from disk
        result1 = cache2.get('key1')
        result2 = cache2.get('key2')

        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
        self.assertEqual(result1.data, 'value1')
        self.assertEqual(result2.data, 'value2')

    def test_delete(self):
        """Test cache deletion"""
        self.cache.set('key1', 'value1')
        self.assertTrue(self.cache.delete('key1'))
        self.assertIsNone(self.cache.get('key1'))

        # Deleting non-existent key
        self.assertFalse(self.cache.delete('nonexistent'))

    def test_clear(self):
        """Test cache clearing"""
        self.cache.set('key1', 'value1')
        self.cache.set('key2', 'value2')

        self.cache.clear()

        self.assertIsNone(self.cache.get('key1'))
        self.assertIsNone(self.cache.get('key2'))
        self.assertEqual(len(self.cache.index), 0)

    def test_exists(self):
        """Test key existence check"""
        self.cache.set('key1', 'value1')
        self.assertTrue(self.cache.exists('key1'))
        self.assertFalse(self.cache.exists('nonexistent'))

    def test_get_all(self):
        """Test getting all entries"""
        self.cache.set('key1', 'value1')
        self.cache.set('key2', 'value2')
        self.cache.set('key3', 'value3')

        all_entries = self.cache.get_all()

        self.assertEqual(len(all_entries), 3)
        self.assertIn('key1', all_entries)
        self.assertIn('key2', all_entries)
        self.assertIn('key3', all_entries)

    def test_cleanup_expired(self):
        """Test cleanup of expired entries"""
        self.cache.set('key1', 'value1')
        self.cache.set('key2', 'value2')

        # Wait for expiration
        time.sleep(2.5)

        # Add new entry (shouldn't expire)
        self.cache.set('key3', 'value3')

        # Cleanup
        removed = self.cache.cleanup_expired()

        self.assertEqual(removed, 2)  # key1 and key2 expired
        self.assertIsNone(self.cache.get('key1'))
        self.assertIsNone(self.cache.get('key2'))
        self.assertIsNotNone(self.cache.get('key3'))

    def test_stats(self):
        """Test cache statistics"""
        self.cache.set('key1', 'value1')
        self.cache.set('key2', 'value2')

        stats = self.cache.get_stats()

        self.assertEqual(stats['total_entries'], 2)
        self.assertGreater(stats['cache_size_mb'], 0)
        self.assertEqual(stats['ttl_seconds'], 2)

    def test_metadata(self):
        """Test entry metadata"""
        metadata = {'source': 'test', 'version': 1}
        self.cache.set('key1', 'value1', metadata=metadata)

        result = self.cache.get('key1')

        self.assertIsNotNone(result)
        self.assertEqual(result.metadata['source'], 'test')
        self.assertEqual(result.metadata['version'], 1)


class TestCacheUtils(unittest.TestCase):
    """Test cache utility functions"""

    def test_compute_hash(self):
        """Test hash computation"""
        hash1 = compute_hash("test")
        hash2 = compute_hash("test")
        hash3 = compute_hash("different")

        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)
        self.assertEqual(len(hash1), 16)

    def test_compute_hash_custom_length(self):
        """Test hash with custom length"""
        hash1 = compute_hash("test", length=8)
        self.assertEqual(len(hash1), 8)

    def test_compute_hash_bytes(self):
        """Test hash with bytes input"""
        hash1 = compute_hash(b"test")
        hash2 = compute_hash("test")
        self.assertEqual(hash1, hash2)

    def test_normalize_path(self):
        """Test path normalization"""
        # Test with backslashes
        path1 = normalize_path("C:\\Users\\test\\file.txt")
        self.assertNotIn('\\', path1)

        # Test with forward slashes
        path2 = normalize_path("/home/user/file.txt")
        self.assertNotIn('\\', path2)

        # Test consistency
        same_path_1 = normalize_path("./test.txt")
        same_path_2 = normalize_path("./test.txt")
        self.assertEqual(same_path_1, same_path_2)

    def test_batch_hash(self):
        """Test batch hashing"""
        items1 = ["item1", "item2", "item3"]
        items2 = ["item3", "item1", "item2"]  # Different order

        hash1 = batch_hash(items1)
        hash2 = batch_hash(items2)

        # Should be equal (sorted before hashing)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 16)


class TestLocationCache(unittest.TestCase):
    """Test LocationCache migration"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_location_cache_structure(self):
        """Test LocationCache default structure"""
        from src.location_service import LocationCache

        cache = LocationCache(cache_dir=Path(self.temp_dir))

        # Should have both sections
        self.assertIn('locations', cache.index)
        self.assertIn('disambiguations', cache.index)
        self.assertEqual(len(cache.index['locations']), 0)
        self.assertEqual(len(cache.index['disambiguations']), 0)

    def test_location_cache_entry_count(self):
        """Test entry counting across sections"""
        from src.location_service import LocationCache

        cache = LocationCache(cache_dir=Path(self.temp_dir))

        # Add entries to both sections
        cache.index['locations']['paris'] = {'data': 'test1', 'cached_at': time.time()}
        cache.index['disambiguations']['ctx1'] = {'data': 'test2', 'cached_at': time.time()}

        # Count should be sum of both
        self.assertEqual(cache._count_entries(), 2)


if __name__ == '__main__':
    unittest.main()
