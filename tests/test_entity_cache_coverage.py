"""
Test coverage for src/entity_cache.py

Target: Cover all 32 missed lines to achieve 100% coverage.

Covers:
- _serialize_entry method
- _deserialize_entry legacy format and unknown format
- _load_index with old 'entities' key migration
- _load_index JSON error handling
- find_entity type mismatch
- _is_valid date parse error
- add_entity copy failure
- get_images_for_project symlink fallback
- cleanup when disabled
"""

import sys
from pathlib import Path
import json
import shutil
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock, Mock


def create_mock_config(enabled=True, cache_dir=None, fuzzy_threshold=0.85,
                       max_age_days=0, cache_strategy='copy'):
    """Create a mock EntityCacheConfig."""
    config = Mock()
    config.enabled = enabled
    config.cache_dir = cache_dir or '~/.test_entity_cache'
    config.fuzzy_threshold = fuzzy_threshold
    config.max_age_days = max_age_days
    config.cache_strategy = cache_strategy
    return config


class TestCachedEntityDataclass:
    """Test CachedEntity dataclass methods."""

    def test_to_dict_returns_dict(self):
        """Test CachedEntity.to_dict() method."""
        from src.entity_cache import CachedEntity

        entity = CachedEntity(
            entity_name="Test",
            entity_type="GPE",
            images=["/path/img1.jpg", "/path/img2.jpg"],
            source_project="TestProject",
            cached_at="2026-01-10T12:00:00",
            query="test query"
        )

        result = entity.to_dict()

        assert isinstance(result, dict)
        assert result['entity_name'] == "Test"
        assert result['entity_type'] == "GPE"
        assert len(result['images']) == 2

    def test_from_dict_creates_entity(self):
        """Test CachedEntity.from_dict() method."""
        from src.entity_cache import CachedEntity

        data = {
            'entity_name': 'Paris',
            'entity_type': 'GPE',
            'images': ['/img1.jpg'],
            'source_project': 'Project1',
            'cached_at': '2026-01-10T12:00:00',
            'query': 'paris city'
        }

        entity = CachedEntity.from_dict(data)

        assert entity.entity_name == 'Paris'
        assert entity.entity_type == 'GPE'
        assert entity.query == 'paris city'


class TestEntityCacheSerializeMethods:
    """Test _serialize_entry and _deserialize_entry methods."""

    def test_serialize_entry(self, tmp_path):
        """Test _serialize_entry converts CacheEntry to dict."""
        from src.entity_cache import EntityCache
        from src.cache import CacheEntry

        config = create_mock_config(cache_dir=str(tmp_path / "cache"))
        cache = EntityCache(config)

        entry = CacheEntry(
            data={'entity_name': 'Test', 'entity_type': 'GPE'},
            cached_at='2026-01-10T12:00:00',
            key='test_key',
            metadata={'source': 'google'}
        )

        result = cache._serialize_entry(entry)

        assert 'data' in result
        assert 'cached_at' in result
        assert 'metadata' in result
        assert result['data']['entity_name'] == 'Test'

    def test_deserialize_entry_new_format(self, tmp_path):
        """Test _deserialize_entry with new BaseCache format."""
        from src.entity_cache import EntityCache

        config = create_mock_config(cache_dir=str(tmp_path / "cache"))
        cache = EntityCache(config)

        data = {
            'data': {'entity_name': 'Test', 'entity_type': 'GPE'},
            'cached_at': '2026-01-10T12:00:00',
            'metadata': {'source': 'bing'}
        }

        entry = cache._deserialize_entry(data)

        assert entry.data['entity_name'] == 'Test'
        assert entry.cached_at == '2026-01-10T12:00:00'

    def test_deserialize_entry_legacy_format(self, tmp_path):
        """Test _deserialize_entry with legacy format (entity_name at top level)."""
        from src.entity_cache import EntityCache

        config = create_mock_config(cache_dir=str(tmp_path / "cache"))
        cache = EntityCache(config)

        # Legacy format: entity data directly (no 'data' wrapper)
        data = {
            'entity_name': 'LegacyEntity',
            'entity_type': 'PERSON',
            'images': ['/old/img.jpg'],
            'source_project': 'OldProject',
            'cached_at': '2025-12-01T12:00:00',
            'query': 'legacy query'
        }

        entry = cache._deserialize_entry(data)

        assert entry.data == data
        assert entry.cached_at == '2025-12-01T12:00:00'

    def test_deserialize_entry_unknown_format_logs_warning(self, tmp_path, caplog):
        """Test _deserialize_entry with unknown format returns empty entry."""
        from src.entity_cache import EntityCache
        import logging

        config = create_mock_config(cache_dir=str(tmp_path / "cache"))
        cache = EntityCache(config)

        # Unknown format - no 'data' and no 'entity_name'
        data = {
            'some_unknown_field': 'value',
            'another_field': 123
        }

        with caplog.at_level(logging.WARNING):
            entry = cache._deserialize_entry(data)

        assert entry.data == {}
        assert "Unknown cache entry format" in caplog.text


class TestEntityCacheLoadIndex:
    """Test _load_index with various scenarios."""

    def test_load_index_old_format_migration(self, tmp_path):
        """Test _load_index migrates old 'entities' format to new format."""
        from src.entity_cache import EntityCache

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        index_path = cache_dir / "entity_cache_index.json"

        # Write old format with 'entities' key
        old_data = {
            'entities': {
                'Paris': {
                    'entity_name': 'Paris',
                    'entity_type': 'GPE',
                    'images': ['/old/paris.jpg'],
                    'source_project': 'OldProject',
                    'cached_at': '2025-06-01T12:00:00',
                    'query': 'paris france'
                }
            }
        }

        with open(index_path, 'w') as f:
            json.dump(old_data, f)

        config = create_mock_config(cache_dir=str(cache_dir))
        cache = EntityCache(config)

        # Check migration occurred
        assert 'Paris' in cache.index
        assert 'data' in cache.index['Paris']  # New format has 'data' wrapper

    def test_load_index_json_error_starts_fresh(self, tmp_path, caplog):
        """Test _load_index starts fresh on JSON decode error."""
        from src.entity_cache import EntityCache
        import logging

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        index_path = cache_dir / "entity_cache_index.json"

        # Write invalid JSON
        index_path.write_text("{ invalid json content [}")

        config = create_mock_config(cache_dir=str(cache_dir))

        with caplog.at_level(logging.WARNING):
            cache = EntityCache(config)

        assert cache.index == {}
        assert "Failed to load cache index" in caplog.text


class TestEntityCacheFindEntity:
    """Test find_entity with various matching scenarios."""

    def test_find_entity_type_mismatch_skipped(self, tmp_path):
        """Test that entities with wrong type are skipped."""
        from src.entity_cache import EntityCache, CachedEntity

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "images").mkdir()

        config = create_mock_config(cache_dir=str(cache_dir))
        cache = EntityCache(config)

        # Create a test image file so _is_valid passes
        img_file = cache_dir / "images" / "test.jpg"
        img_file.write_bytes(b"fake image data")

        # Add entity with type GPE
        cache.set('TestEntity', {
            'entity_name': 'TestEntity',
            'entity_type': 'GPE',
            'images': [str(img_file)],
            'source_project': 'Test',
            'cached_at': datetime.now().isoformat(),
            'query': 'test'
        })

        # Search for PERSON type - should not find GPE entity
        result = cache.find_entity('TestEntity', entity_type='PERSON')

        assert result is None


class TestEntityCacheIsValid:
    """Test _is_valid with edge cases."""

    def test_is_valid_date_parse_error_considers_valid(self, tmp_path):
        """Test that invalid date format is considered valid."""
        from src.entity_cache import EntityCache, CachedEntity

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "images").mkdir()

        config = create_mock_config(cache_dir=str(cache_dir), max_age_days=30)
        cache = EntityCache(config)

        # Create a test image file
        img_file = cache_dir / "images" / "test.jpg"
        img_file.write_bytes(b"fake image data")

        # Create entity with invalid date format
        entity = CachedEntity(
            entity_name='Test',
            entity_type='GPE',
            images=[str(img_file)],
            source_project='Test',
            cached_at='invalid-date-format',  # Invalid!
            query='test'
        )

        # Should return True because it skips the date check on ValueError
        result = cache._is_valid(entity)

        assert result is True


class TestEntityCacheAddEntity:
    """Test add_entity with various scenarios."""

    def test_add_entity_copy_failure_continues(self, tmp_path, caplog):
        """Test that copy failure logs warning and continues."""
        from src.entity_cache import EntityCache
        import logging

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "images").mkdir()

        config = create_mock_config(cache_dir=str(cache_dir))
        cache = EntityCache(config)

        # Create source files
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        img_file = src_dir / "test.jpg"
        img_file.write_bytes(b"fake image data")

        # Make shutil.copy2 fail
        with patch('shutil.copy2', side_effect=Exception("Copy failed")):
            with caplog.at_level(logging.WARNING):
                cache.add_entity(
                    entity_name='FailedCopy',
                    entity_type='GPE',
                    image_paths=[str(img_file)],
                    source_project='Test',
                    query='test'
                )

        assert "Failed to cache image" in caplog.text

    def test_add_entity_disabled_does_nothing(self, tmp_path):
        """Test that add_entity does nothing when cache disabled."""
        from src.entity_cache import EntityCache

        config = create_mock_config(enabled=False, cache_dir=str(tmp_path / "cache"))
        cache = EntityCache(config)

        # Should return early without error
        cache.add_entity(
            entity_name='Test',
            entity_type='GPE',
            image_paths=['/fake/path.jpg'],
            source_project='Test'
        )

        # No entries should be added
        assert len(cache.index) == 0


class TestEntityCacheGetImagesForProject:
    """Test get_images_for_project with various strategies."""

    def test_symlink_fallback_to_copy(self, tmp_path):
        """Test that symlink failure falls back to copy."""
        from src.entity_cache import EntityCache, CachedEntity

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        images_dir = cache_dir / "images"
        images_dir.mkdir()

        # Create cache image file
        cache_img = images_dir / "test.jpg"
        cache_img.write_bytes(b"fake image data")

        config = create_mock_config(
            cache_dir=str(cache_dir),
            cache_strategy='symlink'
        )
        cache = EntityCache(config)

        entity = CachedEntity(
            entity_name='Test',
            entity_type='GPE',
            images=[str(cache_img)],
            source_project='Test',
            cached_at=datetime.now().isoformat(),
            query='test'
        )

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Mock symlink_to to raise OSError
        with patch.object(Path, 'symlink_to', side_effect=OSError("Admin required")):
            result = cache.get_images_for_project(entity, str(project_dir))

        # Should have fallen back to copy
        assert len(result) == 1
        assert Path(result[0]).exists()
        assert Path(result[0]).read_bytes() == b"fake image data"

    def test_reference_strategy_returns_absolute_paths(self, tmp_path):
        """Test that 'reference' strategy returns absolute cache paths."""
        from src.entity_cache import EntityCache, CachedEntity

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        images_dir = cache_dir / "images"
        images_dir.mkdir()

        cache_img = images_dir / "test.jpg"
        cache_img.write_bytes(b"fake image data")

        config = create_mock_config(
            cache_dir=str(cache_dir),
            cache_strategy='reference'
        )
        cache = EntityCache(config)

        entity = CachedEntity(
            entity_name='Test',
            entity_type='GPE',
            images=[str(cache_img)],
            source_project='Test',
            cached_at=datetime.now().isoformat(),
            query='test'
        )

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        result = cache.get_images_for_project(entity, str(project_dir))

        # Should return absolute path to cache (not copied)
        assert len(result) == 1
        assert str(cache_img.resolve()) in result[0]


class TestEntityCacheCleanup:
    """Test cleanup method."""

    def test_cleanup_disabled_returns_zero(self, tmp_path):
        """Test that cleanup returns 0 when cache disabled."""
        from src.entity_cache import EntityCache

        config = create_mock_config(enabled=False, cache_dir=str(tmp_path / "cache"))
        cache = EntityCache(config)

        result = cache.cleanup()

        assert result == 0


class TestEntityCacheGetStats:
    """Test get_stats method."""

    def test_get_stats_disabled_returns_minimal(self, tmp_path):
        """Test that get_stats returns minimal info when disabled."""
        from src.entity_cache import EntityCache

        config = create_mock_config(enabled=False, cache_dir=str(tmp_path / "cache"))
        cache = EntityCache(config)

        stats = cache.get_stats()

        assert stats['enabled'] is False
        assert len(stats) == 1  # Only 'enabled' key


class TestEntityCacheSafeName:
    """Test _safe_name method."""

    def test_safe_name_removes_special_chars(self, tmp_path):
        """Test that _safe_name removes special characters."""
        from src.entity_cache import EntityCache

        config = create_mock_config(cache_dir=str(tmp_path / "cache"))
        cache = EntityCache(config)

        result = cache._safe_name("Test Entity: With/Special\\Chars!")

        assert '/' not in result
        assert '\\' not in result
        assert ':' not in result
        assert '!' not in result

    def test_safe_name_limits_length(self, tmp_path):
        """Test that _safe_name limits string length to 50."""
        from src.entity_cache import EntityCache

        config = create_mock_config(cache_dir=str(tmp_path / "cache"))
        cache = EntityCache(config)

        long_name = "A" * 100
        result = cache._safe_name(long_name)

        assert len(result) <= 50


class TestEntityCacheNewFormatLoading:
    """Test loading already-new-format cache index (lines 147-149)."""

    def test_load_index_new_format_direct(self, tmp_path):
        """Test _load_index loads new format directly without migration."""
        from src.entity_cache import EntityCache

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        index_path = cache_dir / "entity_cache_index.json"

        # Write new format (no 'entities' wrapper)
        new_data = {
            'London': {
                'data': {
                    'entity_name': 'London',
                    'entity_type': 'GPE',
                    'images': ['/new/london.jpg'],
                    'source_project': 'NewProject',
                    'cached_at': '2026-01-10T12:00:00',
                    'query': 'london city'
                },
                'cached_at': '2026-01-10T12:00:00',
                'metadata': {}
            }
        }

        with open(index_path, 'w') as f:
            json.dump(new_data, f)

        config = create_mock_config(cache_dir=str(cache_dir))
        cache = EntityCache(config)

        # Should load directly without migration
        assert 'London' in cache.index
        assert cache.index['London']['data']['entity_name'] == 'London'


class TestEntityCacheGetImagesForProjectEdgeCases:
    """Test get_images_for_project edge cases."""

    def test_get_images_skips_nonexistent_cache_files(self, tmp_path):
        """Test that missing cache files are skipped (line 333)."""
        from src.entity_cache import EntityCache, CachedEntity

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "images").mkdir()

        # Create one real file and reference one nonexistent
        real_img = cache_dir / "images" / "real.jpg"
        real_img.write_bytes(b"real image data")

        config = create_mock_config(
            cache_dir=str(cache_dir),
            cache_strategy='copy'
        )
        cache = EntityCache(config)

        entity = CachedEntity(
            entity_name='Test',
            entity_type='GPE',
            images=[
                str(cache_dir / "images" / "nonexistent.jpg"),  # Doesn't exist
                str(real_img)
            ],
            source_project='Test',
            cached_at=datetime.now().isoformat(),
            query='test'
        )

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        result = cache.get_images_for_project(entity, str(project_dir))

        # Should only have the real file
        assert len(result) == 1
        assert "real.jpg" in result[0]

    def test_get_images_copy_failure_continues(self, tmp_path, caplog):
        """Test that copy failure skips file and continues (lines 356-358)."""
        from src.entity_cache import EntityCache, CachedEntity
        import logging

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        images_dir = cache_dir / "images"
        images_dir.mkdir()

        # Create two cache image files
        cache_img1 = images_dir / "test1.jpg"
        cache_img1.write_bytes(b"fake image data 1")
        cache_img2 = images_dir / "test2.jpg"
        cache_img2.write_bytes(b"fake image data 2")

        config = create_mock_config(
            cache_dir=str(cache_dir),
            cache_strategy='copy'
        )
        cache = EntityCache(config)

        entity = CachedEntity(
            entity_name='Test',
            entity_type='GPE',
            images=[str(cache_img1), str(cache_img2)],
            source_project='Test',
            cached_at=datetime.now().isoformat(),
            query='test'
        )

        project_dir = tmp_path / "project"
        project_dir.mkdir()

        # Make first copy fail, second succeed
        original_copy = shutil.copy2
        call_count = [0]

        def mock_copy(src, dst):
            call_count[0] += 1
            if call_count[0] == 1:
                raise PermissionError("Copy failed")
            return original_copy(src, dst)

        with patch('shutil.copy2', side_effect=mock_copy):
            with caplog.at_level(logging.WARNING):
                result = cache.get_images_for_project(entity, str(project_dir))

        # Should have logged warning
        assert "Failed to copy cached image" in caplog.text
        # Should only have the second file (first copy failed)
        assert len(result) == 1
        assert "test2.jpg" in result[0]


class TestEntityCacheFuzzyMatching:
    """Test fuzzy matching edge cases."""

    def test_fuzzy_similarity_empty_strings(self, tmp_path):
        """Test _fuzzy_similarity with empty strings (line 215-216)."""
        from src.entity_cache import EntityCache

        config = create_mock_config(cache_dir=str(tmp_path / "cache"))
        cache = EntityCache(config)

        assert cache._fuzzy_similarity("", "test") == 0.0
        assert cache._fuzzy_similarity("test", "") == 0.0
        # Two empty strings are identical, so similarity is 1.0
        assert cache._fuzzy_similarity("", "") == 1.0

    def test_fuzzy_similarity_identical_strings(self, tmp_path):
        """Test _fuzzy_similarity with identical strings."""
        from src.entity_cache import EntityCache

        config = create_mock_config(cache_dir=str(tmp_path / "cache"))
        cache = EntityCache(config)

        assert cache._fuzzy_similarity("test", "test") == 1.0
        assert cache._fuzzy_similarity("TEST", "test") == 1.0  # Case insensitive

    def test_find_entity_fuzzy_match_below_threshold(self, tmp_path):
        """Test that fuzzy matches below threshold are not returned."""
        from src.entity_cache import EntityCache

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "images").mkdir()

        # Create a test image
        img_file = cache_dir / "images" / "test.jpg"
        img_file.write_bytes(b"fake image data")

        config = create_mock_config(
            cache_dir=str(cache_dir),
            fuzzy_threshold=0.95  # Very high threshold
        )
        cache = EntityCache(config)

        cache.set('New York City', {
            'entity_name': 'New York City',
            'entity_type': 'GPE',
            'images': [str(img_file)],
            'source_project': 'Test',
            'cached_at': datetime.now().isoformat(),
            'query': 'new york'
        })

        # Should not match because similarity is below 0.95
        result = cache.find_entity('NYC')

        assert result is None


class TestEntityCacheCleanupRemovesInvalid:
    """Test cleanup removes invalid entries."""

    def test_cleanup_removes_expired_entries(self, tmp_path):
        """Test that cleanup removes entries with missing files."""
        from src.entity_cache import EntityCache

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "images").mkdir()

        config = create_mock_config(cache_dir=str(cache_dir))
        cache = EntityCache(config)

        # Add entity with non-existent files
        cache.set('InvalidEntity', {
            'entity_name': 'InvalidEntity',
            'entity_type': 'GPE',
            'images': ['/nonexistent/path.jpg'],  # Doesn't exist
            'source_project': 'Test',
            'cached_at': datetime.now().isoformat(),
            'query': 'test'
        })

        assert 'InvalidEntity' in cache.index

        removed_count = cache.cleanup()

        assert removed_count == 1
        assert 'InvalidEntity' not in cache.index
