"""
Tests for cache management CLI commands.

Tests the CacheManager class and CLI argument handling for:
- --cache-stats
- --cache-clear
- --cache-cleanup
- --cache-list
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import dataclass

from src.cli.cache_commands import (
    CacheManager,
    CacheStats,
    ClearResult,
    CleanupResult,
    display_cache_stats,
    display_cache_list,
)


@dataclass
class MockCacheConfig:
    cache_dir: str = ".cache"


@dataclass
class MockGlobalCacheConfig:
    enabled: bool = True
    cache_dir: str = "~/.matcher_global_cache"


@dataclass
class MockEntityCacheConfig:
    enabled: bool = True
    cache_dir: str = "~/.matcher_entity_cache"


@dataclass
class MockImageSearchConfig:
    entity_cache: MockEntityCacheConfig = None

    def __post_init__(self):
        if self.entity_cache is None:
            self.entity_cache = MockEntityCacheConfig()


@dataclass
class MockConfig:
    cache: MockCacheConfig = None
    global_cache: MockGlobalCacheConfig = None
    image_search: MockImageSearchConfig = None
    llm: Mock = None

    def __post_init__(self):
        if self.cache is None:
            self.cache = MockCacheConfig()
        if self.global_cache is None:
            self.global_cache = MockGlobalCacheConfig()
        if self.image_search is None:
            self.image_search = MockImageSearchConfig()


class TestCacheManagerInit:
    """Tests for CacheManager initialization"""

    @pytest.mark.fast
    def test_init_resolves_paths(self, tmp_path):
        """CacheManager resolves all cache paths"""
        config = MockConfig()
        config.cache.cache_dir = str(tmp_path / ".cache")
        config.global_cache.cache_dir = str(tmp_path / "global")
        config.image_search.entity_cache.cache_dir = str(tmp_path / "entity")

        mgr = CacheManager(config, tmp_path)

        assert 'global' in mgr._cache_paths
        assert 'entity' in mgr._cache_paths
        assert 'transcripts' in mgr._cache_paths
        assert 'llm' in mgr._cache_paths

    @pytest.mark.fast
    def test_init_handles_relative_cache_dir(self, tmp_path):
        """Handles relative cache directory in config"""
        config = MockConfig()
        config.cache.cache_dir = ".cache"

        mgr = CacheManager(config, tmp_path)

        assert mgr._cache_paths['transcripts'] == tmp_path / ".cache" / "transcriptions"


class TestGetAllStats:
    """Tests for get_all_stats method"""

    @pytest.mark.fast
    def test_returns_stats_for_all_cache_types(self, tmp_path):
        """Returns stats for all configured cache types"""
        config = MockConfig()
        config.cache.cache_dir = str(tmp_path / ".cache")
        config.global_cache.cache_dir = str(tmp_path / "global")
        config.image_search.entity_cache.cache_dir = str(tmp_path / "entity")

        mgr = CacheManager(config, tmp_path)
        stats = mgr.get_all_stats()

        assert 'global' in stats
        assert 'entity' in stats
        assert 'transcripts' in stats
        assert all(isinstance(s, CacheStats) for s in stats.values())

    @pytest.mark.fast
    def test_counts_files_in_cache(self, tmp_path):
        """Correctly counts files in cache directories"""
        # Setup cache directory with files
        cache_dir = tmp_path / ".cache" / "llm_responses"
        cache_dir.mkdir(parents=True)
        (cache_dir / "test1.json").write_text('{}')
        (cache_dir / "test2.json").write_text('{}')

        config = MockConfig()
        config.cache.cache_dir = str(tmp_path / ".cache")
        config.global_cache.cache_dir = str(tmp_path / "global")
        config.image_search.entity_cache.cache_dir = str(tmp_path / "entity")

        mgr = CacheManager(config, tmp_path)
        stats = mgr.get_all_stats()

        assert stats['llm'].total_entries == 2

    @pytest.mark.fast
    def test_reports_not_created_status(self, tmp_path):
        """Reports 'not_created' for nonexistent directories"""
        config = MockConfig()
        config.cache.cache_dir = str(tmp_path / "nonexistent" / ".cache")
        config.global_cache.cache_dir = str(tmp_path / "nonexistent_global")
        config.image_search.entity_cache.cache_dir = str(tmp_path / "nonexistent_entity")

        mgr = CacheManager(config, tmp_path)
        stats = mgr.get_all_stats()

        assert stats['transcripts'].additional_info.get('status') == 'not_created'


class TestClearCache:
    """Tests for clear_cache method"""

    @pytest.mark.fast
    def test_clears_single_cache_type(self, tmp_path):
        """Clears only the specified cache type"""
        # Setup cache with files
        llm_dir = tmp_path / ".cache" / "llm_responses"
        llm_dir.mkdir(parents=True)
        (llm_dir / "test.json").write_text('{}')

        trans_dir = tmp_path / ".cache" / "transcriptions"
        trans_dir.mkdir(parents=True)
        (trans_dir / "test.json").write_text('{}')

        config = MockConfig()
        config.cache.cache_dir = str(tmp_path / ".cache")
        config.global_cache.cache_dir = str(tmp_path / "global")
        config.image_search.entity_cache.cache_dir = str(tmp_path / "entity")

        mgr = CacheManager(config, tmp_path)
        result = mgr.clear_cache('llm')

        assert result.entries_removed == 1
        assert not list(llm_dir.glob("*"))  # LLM dir should be empty
        assert list(trans_dir.glob("*"))  # Transcripts should still have files

    @pytest.mark.fast
    def test_clears_all_caches(self, tmp_path):
        """Clears all caches when type is 'all'"""
        # Setup caches
        for name in ['llm_responses', 'transcriptions', 'embeddings']:
            cache_dir = tmp_path / ".cache" / name
            cache_dir.mkdir(parents=True)
            (cache_dir / "test.json").write_text('{}')

        config = MockConfig()
        config.cache.cache_dir = str(tmp_path / ".cache")
        config.global_cache.cache_dir = str(tmp_path / "global")
        config.image_search.entity_cache.cache_dir = str(tmp_path / "entity")

        mgr = CacheManager(config, tmp_path)
        result = mgr.clear_cache('all')

        assert result.entries_removed >= 3
        assert result.cache_type == 'all'

    @pytest.mark.fast
    def test_dry_run_does_not_delete(self, tmp_path):
        """Dry run reports but doesn't delete"""
        llm_dir = tmp_path / ".cache" / "llm_responses"
        llm_dir.mkdir(parents=True)
        test_file = llm_dir / "test.json"
        test_file.write_text('{}')

        config = MockConfig()
        config.cache.cache_dir = str(tmp_path / ".cache")
        config.global_cache.cache_dir = str(tmp_path / "global")
        config.image_search.entity_cache.cache_dir = str(tmp_path / "entity")

        mgr = CacheManager(config, tmp_path)
        result = mgr.clear_cache('llm', dry_run=True)

        assert result.entries_removed == 1
        assert result.dry_run is True
        assert test_file.exists()  # File should still exist

    @pytest.mark.fast
    def test_returns_error_for_unknown_type(self, tmp_path):
        """Returns error for unknown cache type"""
        config = MockConfig()
        config.cache.cache_dir = str(tmp_path / ".cache")
        config.global_cache.cache_dir = str(tmp_path / "global")
        config.image_search.entity_cache.cache_dir = str(tmp_path / "entity")

        mgr = CacheManager(config, tmp_path)
        result = mgr.clear_cache('unknown_type')

        assert len(result.errors) > 0
        assert 'unknown_type' in result.errors[0].lower()


class TestCleanupAll:
    """Tests for cleanup_all method"""

    @pytest.mark.fast
    def test_returns_cleanup_result(self, tmp_path):
        """Returns CleanupResult with details"""
        config = MockConfig()
        config.cache.cache_dir = str(tmp_path / ".cache")
        config.global_cache.cache_dir = str(tmp_path / "global")
        config.image_search.entity_cache.cache_dir = str(tmp_path / "entity")

        mgr = CacheManager(config, tmp_path)
        result = mgr.cleanup_all()

        assert isinstance(result, CleanupResult)
        assert hasattr(result, 'expired_removed')
        assert hasattr(result, 'orphaned_removed')

    @pytest.mark.fast
    def test_dry_run_does_not_modify(self, tmp_path):
        """Dry run doesn't modify caches"""
        config = MockConfig()
        config.cache.cache_dir = str(tmp_path / ".cache")
        config.global_cache.cache_dir = str(tmp_path / "global")
        config.image_search.entity_cache.cache_dir = str(tmp_path / "entity")

        mgr = CacheManager(config, tmp_path)
        result = mgr.cleanup_all(dry_run=True)

        assert result.dry_run is True


class TestListEntries:
    """Tests for list_entries method"""

    @pytest.mark.fast
    def test_list_global_entries(self, tmp_path):
        """Lists entries from global cache"""
        # Setup global cache with entry
        registry_dir = tmp_path / "global" / "video_registry"
        registry_dir.mkdir(parents=True)
        (registry_dir / "abc123.json").write_text(json.dumps({
            'filename': 'test_video.mp4',
            'duration': 60,
            'topics': ['nature'],
            'keywords': ['sunset'],
            'file_exists': True,
            'usage_count': 5,
            'current_path': '/path/to/video.mp4',
        }))

        config = MockConfig()
        config.cache.cache_dir = str(tmp_path / ".cache")
        config.global_cache.cache_dir = str(tmp_path / "global")
        config.image_search.entity_cache.cache_dir = str(tmp_path / "entity")

        mgr = CacheManager(config, tmp_path)
        entries = mgr.list_entries('global')

        assert len(entries) == 1
        assert entries[0]['filename'] == 'test_video.mp4'
        assert entries[0]['usage_count'] == 5

    @pytest.mark.fast
    def test_list_entity_entries(self, tmp_path):
        """Lists entries from entity cache"""
        # Setup entity cache
        entity_dir = tmp_path / "entity"
        entity_dir.mkdir(parents=True)
        (entity_dir / "entity_cache_index.json").write_text(json.dumps({
            'john_doe': {
                'data': {
                    'entity_name': 'John Doe',
                    'entity_type': 'PERSON',
                    'images': ['img1.jpg', 'img2.jpg'],
                    'source_project': 'test_project',
                    'cached_at': '2026-01-10T12:00:00',
                }
            }
        }))

        config = MockConfig()
        config.cache.cache_dir = str(tmp_path / ".cache")
        config.global_cache.cache_dir = str(tmp_path / "global")
        config.image_search.entity_cache.cache_dir = str(tmp_path / "entity")

        mgr = CacheManager(config, tmp_path)
        entries = mgr.list_entries('entity')

        assert len(entries) == 1
        assert entries[0]['entity_name'] == 'John Doe'
        assert entries[0]['image_count'] == 2

    @pytest.mark.fast
    def test_returns_empty_for_nonexistent(self, tmp_path):
        """Returns empty list for nonexistent cache"""
        config = MockConfig()
        config.cache.cache_dir = str(tmp_path / ".cache")
        config.global_cache.cache_dir = str(tmp_path / "nonexistent_global")
        config.image_search.entity_cache.cache_dir = str(tmp_path / "entity")

        mgr = CacheManager(config, tmp_path)
        entries = mgr.list_entries('global')

        assert entries == []


class TestDisplayFunctions:
    """Tests for display helper functions"""

    @pytest.mark.fast
    def test_display_cache_stats_no_error(self, capsys):
        """display_cache_stats doesn't raise"""
        stats = {
            'global': CacheStats(
                name='Global',
                cache_type='global',
                total_entries=10,
                total_size_mb=5.5,
                location='/path/to/cache',
            ),
            'entity': CacheStats(
                name='Entity',
                cache_type='entity',
                total_entries=5,
                total_size_mb=2.0,
                location='/path/to/entity',
            ),
        }

        # Should not raise
        display_cache_stats(stats)

        captured = capsys.readouterr()
        assert 'Global' in captured.out
        assert 'Entity' in captured.out
        assert '10' in captured.out  # entries count

    @pytest.mark.fast
    def test_display_cache_list_no_error(self, capsys):
        """display_cache_list doesn't raise"""
        entries = [
            {
                'filename': 'test.mp4',
                'duration': 60,
                'file_exists': True,
                'usage_count': 3,
                'topics': ['nature'],
                'keywords': ['sunset'],
            }
        ]

        # Should not raise
        display_cache_list(entries, 'global')

        captured = capsys.readouterr()
        assert 'test.mp4' in captured.out


class TestClearResultSummary:
    """Tests for ClearResult.summary method"""

    @pytest.mark.fast
    def test_summary_for_dry_run(self):
        """Summary indicates dry run"""
        result = ClearResult(
            cache_type='llm',
            entries_removed=5,
            bytes_freed=1024,
            dry_run=True
        )

        summary = result.summary()
        assert 'Would remove' in summary
        assert '5' in summary

    @pytest.mark.fast
    def test_summary_for_actual_clear(self):
        """Summary indicates actual clear"""
        result = ClearResult(
            cache_type='llm',
            entries_removed=5,
            bytes_freed=1024,
            dry_run=False
        )

        summary = result.summary()
        assert 'Removed' in summary
        assert '5' in summary


class TestCleanupResultSummary:
    """Tests for CleanupResult.summary method"""

    @pytest.mark.fast
    def test_summary_includes_counts(self):
        """Summary includes expired and orphaned counts"""
        result = CleanupResult(
            expired_removed=3,
            orphaned_removed=2,
            bytes_freed=2048,
            dry_run=False
        )

        summary = result.summary()
        assert '3 expired' in summary
        assert '2 orphaned' in summary
