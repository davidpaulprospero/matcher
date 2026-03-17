#!/usr/bin/env python3
"""Validate caption checkpoint caching is working correctly."""

import json
import tempfile
from pathlib import Path

# Test 1: Verify CheckpointData has caption field
def test_checkpoint_data_structure():
    from src.checkpoint import CheckpointData, CheckpointManager
    
    data = CheckpointData()
    print(f"✓ CheckpointData fields: {list(data.__dataclass_fields__.keys())}")
    
    # Check caption field exists
    assert hasattr(data, 'caption'), "Missing 'caption' field in CheckpointData"
    print(f"✓ 'caption' field exists, default value: {data.caption}")
    
    # Check that caption field can store caption_results
    test_caption_results = {
        'video1': {'video_id': 'video1', 'segments': [], 'language': 'en'},
        'video2': {'video_id': 'video2', 'segments': [], 'language': 'es'}
    }
    
    checkpoint_data = {
        'version': '1.0',
        'created_at': '2024-01-01T00:00:00',
        'updated_at': '2024-01-01T00:00:00',
        'last_completed_stage': 'CAPTION',
        'config_hash': 'abc123',
        'voiceover_path': '',
        'voiceover_hash': '',
        'analyze': {},
        'download': {},
        'caption': {'caption_results': test_caption_results, 'success_count': 2},
        'transcribe': {},
        'match': {},
        'iterative_match': {},
        'download_segments': {}
    }
    
    data = CheckpointData.from_dict(checkpoint_data)
    print(f"✓ Loaded checkpoint with caption data: {len(data.caption)} keys")
    print(f"  - caption_results has {len(data.caption.get('caption_results', {}))} videos")

    # Test passed - do not return True as pytest expects None


# Test 2: Verify CheckpointManager.save and load roundtrip
def test_checkpoint_save_load():
    from src.checkpoint import CheckpointManager, CheckpointData
    
    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = Path(tmpdir)
        
        # Create manager and save caption data
        manager = CheckpointManager(project_dir, config_hash='test')
        
        caption_stage_data = {
            'caption_results': {
                'abc123XYZ_0': {
                    'video_id': 'abc123XYZ_0',
                    'segments': [{'text': 'Hello', 'start': 0.0, 'end': 2.0}],
                    'language': 'en',
                    'is_auto_generated': False,
                    'segment_count': 1,
                }
            },
            'success_count': 1,
            'skip_count': 0
        }
        
        # Simulate what PipelineOrchestrator does
        manager.data = CheckpointData(created_at='2024-01-01T00:00:00')
        manager.save('CAPTION', caption_stage_data)
        
        print(f"✓ Saved checkpoint to {manager.checkpoint_path}")
        print(f"  - File exists: {manager.checkpoint_path.exists()}")
        
        # Create new manager and load
        manager2 = CheckpointManager(project_dir, config_hash='test')
        loaded_data = manager2.load()
        
        print(f"✓ Loaded checkpoint: {loaded_data is not None}")
        if loaded_data:
            print(f"  - caption field: {len(loaded_data.caption)} keys")
            caption_results = loaded_data.caption.get('caption_results', {})
            print(f"  - caption_results: {len(caption_results)} videos")
            
        # Test get_stage_data (what CaptionStage._load_existing_captions uses)
        stage_data = manager2.get_stage_data('CAPTION')
        print(f"✓ get_stage_data('CAPTION') returned: {len(stage_data)} keys")
        
        caption_results = stage_data.get('caption_results', {})
        print(f"  - caption_results: {len(caption_results)} videos")
        
        assert len(caption_results) == 1, f"Expected 1 video, got {len(caption_results)}"
        print("✓ Checkpoint save/load roundtrip works correctly!")
        # Test passed - do not return True as pytest expects None


# Test 3: Verify CaptionStage._load_existing_captions
def test_caption_stage_load_existing():
    from src.stages.caption_stage import CaptionStage
    from src.checkpoint import CheckpointManager, CheckpointData
    from unittest.mock import MagicMock
    
    stage = CaptionStage()
    
    # Test with mock checkpoint that returns caption data
    mock_checkpoint = MagicMock()
    mock_checkpoint.get_stage_data.return_value = {
        'caption_results': {
            'video1': {'video_id': 'video1', 'language': 'en'},
            'video2': {'video_id': 'video2', 'language': 'es'}
        }
    }
    
    result = stage._load_existing_captions(mock_checkpoint)
    print(f"✓ _load_existing_captions returned: {len(result)} videos")
    assert len(result) == 2, f"Expected 2 videos, got {len(result)}"
    
    # Test with None return (no checkpoint data)
    mock_checkpoint.get_stage_data.return_value = None
    result = stage._load_existing_captions(mock_checkpoint)
    print(f"✓ _load_existing_captions with None returned: {len(result)} videos (expected 0)")
    assert len(result) == 0, f"Expected 0 videos, got {len(result)}"
    
    # Test with empty dict return
    mock_checkpoint.get_stage_data.return_value = {}
    result = stage._load_existing_captions(mock_checkpoint)
    print(f"✓ _load_existing_captions with empty dict returned: {len(result)} videos (expected 0)")
    assert len(result) == 0, f"Expected 0 videos, got {len(result)}"
    
    print("✓ CaptionStage._load_existing_captions works correctly!")
    # Test passed - do not return True as pytest expects None


# =============================================================================
# STRESS TESTS (US-34-012)
# =============================================================================

import pytest
import threading
import time as time_module
import os


@pytest.mark.stress
class TestConcurrentCacheAccess:
    """Test concurrent cache reads/writes from 10 threads."""

    def test_concurrent_reads_writes_10_threads(self, tmp_path):
        """Test concurrent cache reads/writes from 10 threads."""
        from src.caption.cache import CaptionCache
        from src.caption.cache_models import CachedCaption

        cache = CaptionCache(config=None)
        cache.cache_dir = tmp_path
        cache.index_path = tmp_path / "caption_cache_index.json"
        cache.index = {}
        cache.enabled = True
        cache.auto_save = False  # Disable auto-save for thread safety

        errors = []
        success_counts = {'reads': 0, 'writes': 0}
        lock = threading.Lock()

        def worker(thread_id):
            """Worker that performs reads and writes."""
            try:
                for i in range(10):
                    video_id = f"vid{thread_id:02d}{i:02d}"
                    language = "en"

                    # Write
                    cached = CachedCaption(
                        video_id=video_id,
                        language=language,
                        segments=[{'text': f'seg{i}', 'start': i * 1.0, 'end': (i + 1) * 1.0}],
                        is_auto_generated=False,
                        format_source="json3",
                        fetch_timestamp=time_module.time(),
                        duration=10.0,
                    )
                    cache.set(cache._make_cache_key(video_id, language), cached.to_dict())
                    with lock:
                        success_counts['writes'] += 1

                    # Read
                    result = cache.get_caption(video_id, language)
                    if result is not None:
                        with lock:
                            success_counts['reads'] += 1
            except Exception as e:
                with lock:
                    errors.append((thread_id, str(e)))

        # Start 10 threads
        threads = []
        for t in range(10):
            thread = threading.Thread(target=worker, args=(t,))
            threads.append(thread)
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join(timeout=30)

        # Verify no errors
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert success_counts['writes'] == 100, f"Expected 100 writes, got {success_counts['writes']}"
        assert success_counts['reads'] >= 50, f"Expected at least 50 reads, got {success_counts['reads']}"

    def test_concurrent_index_updates(self, tmp_path):
        """Test concurrent index updates don't corrupt data."""
        from src.caption.cache import CaptionCache
        from src.caption.cache_models import CachedCaption

        cache = CaptionCache(config=None)
        cache.cache_dir = tmp_path
        cache.index_path = tmp_path / "caption_cache_index.json"
        cache.index = {}
        cache.enabled = True
        cache.auto_save = False

        results = {'final_count': 0}

        def writer(thread_id):
            for i in range(20):
                video_id = f"video_{thread_id}_{i}"
                cached = CachedCaption(
                    video_id=video_id,
                    language="en",
                    segments=[],
                    is_auto_generated=False,
                    format_source="test",
                    fetch_timestamp=time_module.time(),
                )
                cache.set(cache._make_cache_key(video_id, "en"), cached.to_dict())

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # Count non-metadata entries
        entry_count = sum(1 for k in cache.index if not k.startswith('__'))

        # With 10 threads x 20 writes = 200 entries expected
        assert entry_count == 200, f"Expected 200 entries, got {entry_count}"


@pytest.mark.stress
class TestCacheEvictionUnderPressure:
    """Test cache eviction under memory pressure (100+ entries)."""

    def test_eviction_with_100_plus_entries(self, tmp_path):
        """Test cache handles 100+ entries without performance degradation."""
        from src.caption.cache import CaptionCache
        from src.caption.cache_models import CachedCaption

        cache = CaptionCache(config=None)
        cache.cache_dir = tmp_path
        cache.index_path = tmp_path / "caption_cache_index.json"
        cache.index = {}
        cache.enabled = True
        cache.auto_save = False

        # Add 150 entries with varying ages by directly setting cached_at
        for i in range(150):
            video_id = f"evict_video_{i:04d}"
            key = cache._make_cache_key(video_id, "en")
            age_seconds = i * 86400  # 0-150 days old
            cached_at = time_module.time() - age_seconds

            cached = CachedCaption(
                video_id=video_id,
                language="en",
                segments=[{'text': f'segment_{j}', 'start': j, 'end': j + 1} for j in range(10)],
                is_auto_generated=False,
                format_source="json3",
                fetch_timestamp=cached_at,
                duration=100.0,
            )
            # Directly set index entry with historical cached_at
            cache.index[key] = {
                'data': cached.to_dict(),
                'cached_at': cached_at,
                'metadata': {}
            }

        cache._save_index()

        # Verify 150 entries exist
        entry_count = sum(1 for k in cache.index if not k.startswith('__'))
        assert entry_count == 150, f"Expected 150 entries, got {entry_count}"

        # Test cleanup of old entries (entries older than 30 days)
        old_count = 0
        for key in cache.index:
            if key.startswith('__'):
                continue
            try:
                entry = cache._deserialize_entry(cache.index[key])
                if cache.is_stale(entry, max_age_days=30):
                    old_count += 1
            except Exception:
                pass

        # With 150 entries aged 0-150 days, entries 31-150 (120 entries) are stale
        assert old_count >= 100, f"Expected at least 100 stale entries, got {old_count}"

        # Test eviction
        result = cache.cleanup_stale_entries(max_age_days=30, dry_run=False)
        assert result['entries_removed'] >= 100, f"Expected to remove 100+ entries, removed {result['entries_removed']}"

    def test_bulk_write_performance(self, tmp_path):
        """Test bulk write performance with many entries."""
        from src.caption.cache import CaptionCache
        from src.caption.cache_models import CachedCaption

        cache = CaptionCache(config=None)
        cache.cache_dir = tmp_path
        cache.index_path = tmp_path / "caption_cache_index.json"
        cache.index = {}
        cache.enabled = True
        cache.auto_save = False

        start = time_module.perf_counter()

        # Write 200 entries
        for i in range(200):
            video_id = f"perf_video_{i:04d}"
            cached = CachedCaption(
                video_id=video_id,
                language="en",
                segments=[{'text': 'hello', 'start': 0, 'end': 1}],
                is_auto_generated=False,
                format_source="json3",
                fetch_timestamp=time_module.time(),
            )
            cache.set(cache._make_cache_key(video_id, "en"), cached.to_dict())

        elapsed = time_module.perf_counter() - start

        # Should complete in under 5 seconds
        assert elapsed < 5.0, f"Bulk write took {elapsed:.2f}s, expected < 5s"

        # Verify all entries exist
        entry_count = sum(1 for k in cache.index if not k.startswith('__'))
        assert entry_count == 200


@pytest.mark.stress
class TestCacheCorruptionRecovery:
    """Test cache corruption recovery (malformed JSON)."""

    def test_malformed_json_recovery(self, tmp_path):
        """Test recovery from malformed JSON index file."""
        from src.caption.cache import CaptionCache

        # Write malformed JSON to index file
        index_path = tmp_path / "caption_cache_index.json"
        with open(index_path, 'w') as f:
            f.write("{invalid json here")

        # Cache should initialize with empty index
        cache = CaptionCache(config=None)
        cache.cache_dir = tmp_path
        cache.index_path = index_path
        cache._load_index()

        # Should have recovered with empty index
        assert cache.index == {}, "Expected empty index after corruption recovery"

    def test_partial_json_recovery(self, tmp_path):
        """Test recovery from partially corrupted JSON."""
        from src.caption.cache import CaptionCache

        # Write truncated JSON
        index_path = tmp_path / "caption_cache_index.json"
        with open(index_path, 'w') as f:
            f.write('{"video1_en": {"data": {"video_id": "video1"')

        cache = CaptionCache(config=None)
        cache.cache_dir = tmp_path
        cache.index_path = index_path
        cache._load_index()

        assert cache.index == {}, "Expected empty index after partial corruption"

    def test_corrupted_entry_isolation(self, tmp_path):
        """Test that corrupted entries don't affect valid ones."""
        from src.caption.cache import CaptionCache
        from src.caption.cache_models import CachedCaption
        import json

        cache = CaptionCache(config=None)
        cache.cache_dir = tmp_path
        cache.index_path = tmp_path / "caption_cache_index.json"
        cache.index = {}
        cache.enabled = True

        # Add valid entry
        valid_cached = CachedCaption(
            video_id="valid_video",
            language="en",
            segments=[{'text': 'hello', 'start': 0, 'end': 1}],
            is_auto_generated=False,
            format_source="json3",
            fetch_timestamp=time_module.time(),
        )
        cache.set(cache._make_cache_key("valid_video", "en"), valid_cached.to_dict())

        # Manually inject corrupted entry
        cache.index["corrupted_en"] = {
            'data': 'not a valid dict',
            'cached_at': 'not a number',
            'metadata': None
        }
        cache._save_index()

        # Valid entry should still be retrievable
        result = cache.get_caption("valid_video", "en")
        assert result is not None, "Valid entry should be retrievable"
        assert result.video_id == "valid_video"

        # Corrupted entry should return None gracefully
        # Direct access via cache.get() which catches exceptions
        corrupted_entry = cache.get("corrupted_en")
        # May return None due to deserialization failure or invalid data
        # Just ensure no exception is raised

    def test_empty_json_file_recovery(self, tmp_path):
        """Test recovery from empty JSON file."""
        from src.caption.cache import CaptionCache

        index_path = tmp_path / "caption_cache_index.json"
        with open(index_path, 'w') as f:
            f.write('')

        cache = CaptionCache(config=None)
        cache.cache_dir = tmp_path
        cache.index_path = index_path
        cache._load_index()

        assert cache.index == {}


@pytest.mark.stress
class TestCacheStalenessDetection:
    """Test cache staleness detection (modified_time checks)."""

    def test_staleness_detection_by_age(self, tmp_path):
        """Test staleness detection based on entry age."""
        from src.caption.cache import CaptionCache
        from src.caption.cache_models import CachedCaption
        from src.cache import CacheEntry

        cache = CaptionCache(config=None)
        cache.cache_dir = tmp_path
        cache.index_path = tmp_path / "caption_cache_index.json"
        cache.index = {}
        cache.enabled = True
        cache.max_age_days = 7

        # Create entry from 10 days ago
        old_entry = CacheEntry(
            data={
                'video_id': 'old_video',
                'language': 'en',
                'segments': [],
                'is_auto_generated': False,
                'format_source': 'json3',
                'fetch_timestamp': time_module.time() - (10 * 86400),
            },
            cached_at=time_module.time() - (10 * 86400),
            key='old_video_en',
        )

        # Create recent entry
        new_entry = CacheEntry(
            data={
                'video_id': 'new_video',
                'language': 'en',
                'segments': [],
                'is_auto_generated': False,
                'format_source': 'json3',
                'fetch_timestamp': time_module.time(),
            },
            cached_at=time_module.time(),
            key='new_video_en',
        )

        # Old entry should be stale
        assert cache.is_stale(old_entry, max_age_days=7) is True

        # New entry should not be stale
        assert cache.is_stale(new_entry, max_age_days=7) is False

    def test_staleness_with_different_thresholds(self, tmp_path):
        """Test staleness detection with various max_age_days thresholds."""
        from src.caption.cache import CaptionCache
        from src.cache import CacheEntry

        cache = CaptionCache(config=None)
        cache.cache_dir = tmp_path
        cache.index_path = tmp_path / "caption_cache_index.json"
        cache.index = {}
        cache.enabled = True

        # Entry from 5 days ago
        entry_5_days = CacheEntry(
            data={},
            cached_at=time_module.time() - (5 * 86400),
            key='test',
        )

        # Test various thresholds
        assert cache.is_stale(entry_5_days, max_age_days=3) is True
        assert cache.is_stale(entry_5_days, max_age_days=7) is False
        assert cache.is_stale(entry_5_days, max_age_days=0) is False  # 0 = no expiration

    def test_get_entry_age_days(self, tmp_path):
        """Test get_entry_age_days calculation."""
        from src.caption.cache import CaptionCache
        from src.cache import CacheEntry

        cache = CaptionCache(config=None)
        cache.cache_dir = tmp_path
        cache.index_path = tmp_path / "caption_cache_index.json"
        cache.index = {}

        # Entry from exactly 10 days ago
        entry = CacheEntry(
            data={},
            cached_at=time_module.time() - (10 * 86400),
            key='test',
        )

        age = cache.get_entry_age_days(entry)
        # Allow small tolerance for test execution time
        assert 9.9 <= age <= 10.1, f"Expected ~10 days, got {age}"

    def test_validation_mode_strict_rejects_stale(self, tmp_path):
        """Test strict validation mode rejects stale entries."""
        from src.caption.cache import CaptionCache
        from src.caption.cache_models import CachedCaption

        cache = CaptionCache(config=None)
        cache.cache_dir = tmp_path
        cache.index_path = tmp_path / "caption_cache_index.json"
        cache.index = {}
        cache.enabled = True
        cache.max_age_days = 7
        cache.validation_mode = 'strict'

        # Store entry with old timestamp
        video_id = "stale_video"
        cache.index[cache._make_cache_key(video_id, "en")] = {
            'data': {
                'video_id': video_id,
                'language': 'en',
                'segments': [{'text': 'test', 'start': 0, 'end': 1}],
                'is_auto_generated': False,
                'format_source': 'json3',
                'fetch_timestamp': time_module.time() - (10 * 86400),
                'duration': 10.0,
            },
            'cached_at': time_module.time() - (10 * 86400),
            'metadata': {}
        }

        # Strict mode should return None for stale entry
        result = cache.get_caption(video_id, "en")
        assert result is None, "Strict mode should reject stale entries"

    def test_validation_mode_warn_returns_stale(self, tmp_path):
        """Test warn validation mode returns stale entries with warning."""
        from src.caption.cache import CaptionCache

        cache = CaptionCache(config=None)
        cache.cache_dir = tmp_path
        cache.index_path = tmp_path / "caption_cache_index.json"
        cache.index = {}
        cache.enabled = True
        cache.max_age_days = 7
        cache.validation_mode = 'warn'

        video_id = "stale_warn_video"
        cache.index[cache._make_cache_key(video_id, "en")] = {
            'data': {
                'video_id': video_id,
                'language': 'en',
                'segments': [{'text': 'test', 'start': 0, 'end': 1}],
                'is_auto_generated': False,
                'format_source': 'json3',
                'fetch_timestamp': time_module.time() - (10 * 86400),
                'duration': 10.0,
            },
            'cached_at': time_module.time() - (10 * 86400),
            'metadata': {}
        }

        # Warn mode should return stale entry
        result = cache.get_caption(video_id, "en")
        assert result is not None, "Warn mode should return stale entries"
        assert result.video_id == video_id

    def test_validation_mode_skip_ignores_staleness(self, tmp_path):
        """Test skip validation mode ignores staleness check."""
        from src.caption.cache import CaptionCache

        cache = CaptionCache(config=None)
        cache.cache_dir = tmp_path
        cache.index_path = tmp_path / "caption_cache_index.json"
        cache.index = {}
        cache.enabled = True
        cache.max_age_days = 7
        cache.validation_mode = 'skip'

        video_id = "stale_skip_video"
        cache.index[cache._make_cache_key(video_id, "en")] = {
            'data': {
                'video_id': video_id,
                'language': 'en',
                'segments': [{'text': 'test', 'start': 0, 'end': 1}],
                'is_auto_generated': False,
                'format_source': 'json3',
                'fetch_timestamp': time_module.time() - (100 * 86400),  # Very old
                'duration': 10.0,
            },
            'cached_at': time_module.time() - (100 * 86400),
            'metadata': {}
        }

        # Skip mode should return entry regardless of age
        result = cache.get_caption(video_id, "en")
        assert result is not None, "Skip mode should ignore staleness"


if __name__ == '__main__':
    print("=" * 60)
    print("Caption Cache Validation Tests")
    print("=" * 60)

    try:
        print("\n--- Test 1: CheckpointData Structure ---")
        test_checkpoint_data_structure()

        print("\n--- Test 2: Checkpoint Save/Load Roundtrip ---")
        test_checkpoint_save_load()

        print("\n--- Test 3: CaptionStage._load_existing_captions ---")
        test_caption_stage_load_existing()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
