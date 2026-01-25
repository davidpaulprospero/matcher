"""
Tests for CaptionCache in src/caption_fetcher.py

Tests cache hits, misses, invalidation, and TTL expiration.
"""

import json
import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from src.caption_fetcher import (
    CaptionCache,
    CachedCaption,
    CaptionResult,
    CaptionSegment,
    CaptionFetcher,
    CaptionUnavailableError,
)


class TestCachedCaption:
    """Test CachedCaption dataclass"""

    def test_cached_caption_creation(self):
        """Test creating a cached caption"""
        cached = CachedCaption(
            video_id="dQw4w9WgXcQ",
            language="en",
            segments=[
                {'index': 0, 'start': 0.0, 'end': 2.0, 'text': 'Hello', 'source_file': 'dQw4w9WgXcQ'},
                {'index': 1, 'start': 2.0, 'end': 4.0, 'text': 'World', 'source_file': 'dQw4w9WgXcQ'},
            ],
            is_auto_generated=False,
            format_source='vtt',
            fetch_timestamp=1234567890.0,
            duration=4.0
        )

        assert cached.video_id == "dQw4w9WgXcQ"
        assert cached.language == "en"
        assert len(cached.segments) == 2
        assert cached.is_auto_generated is False
        assert cached.format_source == 'vtt'
        assert cached.fetch_timestamp == 1234567890.0
        assert cached.duration == 4.0

    def test_cached_caption_to_dict(self):
        """Test serialization to dict"""
        cached = CachedCaption(
            video_id="test123video",
            language="es",
            segments=[{'index': 0, 'start': 0.0, 'end': 1.0, 'text': 'Hola', 'source_file': 'test123video'}],
            is_auto_generated=True,
            format_source='json3',
            fetch_timestamp=1000000.0,
            duration=1.0
        )

        result = cached.to_dict()

        assert result['video_id'] == "test123video"
        assert result['language'] == "es"
        assert result['segments'] == [{'index': 0, 'start': 0.0, 'end': 1.0, 'text': 'Hola', 'source_file': 'test123video'}]
        assert result['is_auto_generated'] is True
        assert result['format_source'] == 'json3'
        assert result['fetch_timestamp'] == 1000000.0
        assert result['duration'] == 1.0

    def test_cached_caption_from_dict(self):
        """Test deserialization from dict"""
        data = {
            'video_id': 'abc123def45',
            'language': 'fr',
            'segments': [{'index': 0, 'start': 0.0, 'end': 3.0, 'text': 'Bonjour', 'source_file': 'abc123def45'}],
            'is_auto_generated': False,
            'format_source': 'srt',
            'fetch_timestamp': 9999999.0,
            'duration': 3.0
        }

        cached = CachedCaption.from_dict(data)

        assert cached.video_id == 'abc123def45'
        assert cached.language == 'fr'
        assert len(cached.segments) == 1
        assert cached.is_auto_generated is False
        assert cached.format_source == 'srt'
        assert cached.fetch_timestamp == 9999999.0
        assert cached.duration == 3.0

    def test_cached_caption_to_caption_result(self):
        """Test conversion to CaptionResult"""
        cached = CachedCaption(
            video_id="vid12345678",
            language="en",
            segments=[
                {'index': 0, 'start': 0.0, 'end': 2.0, 'text': 'First', 'source_file': 'vid12345678'},
                {'index': 1, 'start': 2.0, 'end': 4.0, 'text': 'Second', 'source_file': 'vid12345678'},
            ],
            is_auto_generated=True,
            format_source='vtt',
            fetch_timestamp=12345.0,
            duration=4.0
        )

        result = cached.to_caption_result()

        assert isinstance(result, CaptionResult)
        assert result.video_id == "vid12345678"
        assert result.language == "en"
        assert len(result.segments) == 2
        assert result.is_auto_generated is True
        assert result.format_source == 'vtt'

        # Check segments converted correctly
        assert result.segments[0].index == 0
        assert result.segments[0].start_time == 0.0
        assert result.segments[0].end_time == 2.0
        assert result.segments[0].text == 'First'
        assert result.segments[1].text == 'Second'


class TestCaptionCacheBasics:
    """Test basic CaptionCache functionality"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory"""
        cache_dir = tmp_path / "test_caption_cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def mock_config(self, temp_cache_dir):
        """Create a mock CaptionFirstConfig"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 30
        config.cache_captions = True
        return config

    @pytest.fixture
    def cache(self, mock_config):
        """Create a CaptionCache instance"""
        return CaptionCache(mock_config)

    def test_cache_initialization(self, mock_config, temp_cache_dir):
        """Test cache initializes correctly"""
        cache = CaptionCache(mock_config)

        assert cache.enabled is True
        assert cache.max_age_days == 30
        assert cache.cache_dir == temp_cache_dir

    def test_cache_initialization_defaults(self, tmp_path):
        """Test cache with no config uses defaults"""
        # Patch expanduser to use temp directory
        with patch('os.path.expanduser', return_value=str(tmp_path / "default_cache")):
            cache = CaptionCache(None)

            assert cache.enabled is True
            assert cache.max_age_days == 30

    def test_cache_disabled(self, temp_cache_dir):
        """Test cache respects enabled=False"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 30
        config.cache_captions = False

        cache = CaptionCache(config)

        assert cache.enabled is False

    def test_make_cache_key(self, cache):
        """Test cache key generation"""
        key = cache._make_cache_key("dQw4w9WgXcQ", "en")
        assert key == "dQw4w9WgXcQ_en"

        key = cache._make_cache_key("abc123def45", "es")
        assert key == "abc123def45_es"


class TestCaptionCacheOperations:
    """Test CaptionCache store/get operations"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory"""
        cache_dir = tmp_path / "test_caption_cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def mock_config(self, temp_cache_dir):
        """Create a mock CaptionFirstConfig"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 30
        config.cache_captions = True
        return config

    @pytest.fixture
    def cache(self, mock_config):
        """Create a CaptionCache instance"""
        return CaptionCache(mock_config)

    @pytest.fixture
    def sample_result(self):
        """Create a sample CaptionResult"""
        segments = [
            CaptionSegment(0, 0.0, 2.0, "Hello world", "dQw4w9WgXcQ"),
            CaptionSegment(1, 2.0, 4.0, "This is a test", "dQw4w9WgXcQ"),
            CaptionSegment(2, 4.0, 6.0, "Caption text here", "dQw4w9WgXcQ"),
        ]
        return CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=segments,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )

    def test_store_and_get(self, cache, sample_result):
        """Test storing and retrieving captions"""
        # Store
        success = cache.store(sample_result)
        assert success is True

        # Get
        cached = cache.get_caption("dQw4w9WgXcQ", "en")
        assert cached is not None
        assert cached.video_id == "dQw4w9WgXcQ"
        assert cached.language == "en"
        assert len(cached.segments) == 3
        assert cached.is_auto_generated is False
        assert cached.format_source == "vtt"

    def test_cache_miss(self, cache):
        """Test cache miss returns None"""
        cached = cache.get_caption("nonexistent1", "en")
        assert cached is None

    def test_cache_miss_different_language(self, cache, sample_result):
        """Test cache miss for different language"""
        cache.store(sample_result)

        # Try to get with different language
        cached = cache.get_caption("dQw4w9WgXcQ", "es")
        assert cached is None

    def test_store_empty_result(self, cache):
        """Test storing empty result is rejected"""
        empty_result = CaptionResult(
            video_id="emptyvideoxx",
            segments=[],
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )

        success = cache.store(empty_result)
        assert success is False

        cached = cache.get_caption("emptyvideoxx", "en")
        assert cached is None

    def test_store_when_disabled(self, temp_cache_dir, sample_result):
        """Test store fails when cache is disabled"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 30
        config.cache_captions = False

        cache = CaptionCache(config)
        success = cache.store(sample_result)

        assert success is False

    def test_get_when_disabled(self, temp_cache_dir, sample_result):
        """Test get returns None when cache is disabled"""
        # First store with enabled cache
        config_enabled = Mock()
        config_enabled.cache_dir = str(temp_cache_dir)
        config_enabled.max_cache_age_days = 30
        config_enabled.cache_captions = True

        cache_enabled = CaptionCache(config_enabled)
        cache_enabled.store(sample_result)

        # Now try to get with disabled cache
        config_disabled = Mock()
        config_disabled.cache_dir = str(temp_cache_dir)
        config_disabled.max_cache_age_days = 30
        config_disabled.cache_captions = False

        cache_disabled = CaptionCache(config_disabled)
        cached = cache_disabled.get_caption("dQw4w9WgXcQ", "en")

        assert cached is None

    def test_multiple_languages(self, cache):
        """Test caching same video in multiple languages"""
        # English version
        en_segments = [CaptionSegment(0, 0.0, 2.0, "Hello", "dQw4w9WgXcQ")]
        en_result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=en_segments,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )
        cache.store(en_result)

        # Spanish version
        es_segments = [CaptionSegment(0, 0.0, 2.0, "Hola", "dQw4w9WgXcQ")]
        es_result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=es_segments,
            language="es",
            is_auto_generated=True,
            format_source="vtt"
        )
        cache.store(es_result)

        # Retrieve both
        cached_en = cache.get_caption("dQw4w9WgXcQ", "en")
        cached_es = cache.get_caption("dQw4w9WgXcQ", "es")

        assert cached_en is not None
        assert cached_es is not None
        assert cached_en.language == "en"
        assert cached_es.language == "es"
        assert cached_en.segments[0]['text'] == "Hello"
        assert cached_es.segments[0]['text'] == "Hola"
        assert cached_en.is_auto_generated is False
        assert cached_es.is_auto_generated is True


class TestCaptionCacheInvalidation:
    """Test CaptionCache invalidation functionality"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory"""
        cache_dir = tmp_path / "test_caption_cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def mock_config(self, temp_cache_dir):
        """Create a mock CaptionFirstConfig"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 30
        config.cache_captions = True
        return config

    @pytest.fixture
    def cache(self, mock_config):
        """Create a CaptionCache instance"""
        return CaptionCache(mock_config)

    def test_invalidate_specific_language(self, cache):
        """Test invalidating a specific video+language"""
        # Store multiple entries
        for lang in ["en", "es", "fr"]:
            segments = [CaptionSegment(0, 0.0, 2.0, f"Text in {lang}", "dQw4w9WgXcQ")]
            result = CaptionResult(
                video_id="dQw4w9WgXcQ",
                segments=segments,
                language=lang,
                is_auto_generated=False,
                format_source="vtt"
            )
            cache.store(result)

        # Invalidate just English
        count = cache.invalidate("dQw4w9WgXcQ", "en")

        assert count == 1
        assert cache.get_caption("dQw4w9WgXcQ", "en") is None
        assert cache.get_caption("dQw4w9WgXcQ", "es") is not None
        assert cache.get_caption("dQw4w9WgXcQ", "fr") is not None

    def test_invalidate_all_languages(self, cache):
        """Test invalidating all languages for a video"""
        # Store multiple entries
        for lang in ["en", "es", "fr"]:
            segments = [CaptionSegment(0, 0.0, 2.0, f"Text in {lang}", "dQw4w9WgXcQ")]
            result = CaptionResult(
                video_id="dQw4w9WgXcQ",
                segments=segments,
                language=lang,
                is_auto_generated=False,
                format_source="vtt"
            )
            cache.store(result)

        # Invalidate all languages
        count = cache.invalidate("dQw4w9WgXcQ")

        assert count == 3
        assert cache.get_caption("dQw4w9WgXcQ", "en") is None
        assert cache.get_caption("dQw4w9WgXcQ", "es") is None
        assert cache.get_caption("dQw4w9WgXcQ", "fr") is None

    def test_invalidate_nonexistent(self, cache):
        """Test invalidating nonexistent entry returns 0"""
        count = cache.invalidate("nonexistent1", "en")
        assert count == 0

    def test_invalidate_doesnt_affect_other_videos(self, cache):
        """Test invalidation doesn't affect other videos"""
        # Store entries for two videos
        for video_id in ["dQw4w9WgXcQ", "abc123def45"]:
            segments = [CaptionSegment(0, 0.0, 2.0, "Test", video_id)]
            result = CaptionResult(
                video_id=video_id,
                segments=segments,
                language="en",
                is_auto_generated=False,
                format_source="vtt"
            )
            cache.store(result)

        # Invalidate first video
        cache.invalidate("dQw4w9WgXcQ")

        # Second video should be unaffected
        assert cache.get_caption("abc123def45", "en") is not None


class TestCaptionCacheTTL:
    """Test CaptionCache TTL/expiration functionality"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory"""
        cache_dir = tmp_path / "test_caption_cache"
        cache_dir.mkdir()
        return cache_dir

    def test_expired_entry_returns_none(self, temp_cache_dir):
        """Test expired entries are not returned"""
        # Use very short TTL (1 second = ~0.00001 days)
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 0.00001  # About 0.8 seconds
        config.cache_captions = True

        cache = CaptionCache(config)

        # Store an entry
        segments = [CaptionSegment(0, 0.0, 2.0, "Test", "dQw4w9WgXcQ")]
        result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=segments,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )
        cache.store(result)

        # Should be available immediately
        assert cache.get_caption("dQw4w9WgXcQ", "en") is not None

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired now
        cached = cache.get_caption("dQw4w9WgXcQ", "en")
        assert cached is None

    def test_no_expiration_with_zero_days(self, temp_cache_dir):
        """Test TTL=0 means no expiration"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 0  # No expiration
        config.cache_captions = True

        cache = CaptionCache(config)

        # Store an entry
        segments = [CaptionSegment(0, 0.0, 2.0, "Test", "dQw4w9WgXcQ")]
        result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=segments,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )
        cache.store(result)

        # Should still be available (no expiration)
        assert cache.get_caption("dQw4w9WgXcQ", "en") is not None


class TestCaptionCacheGetOrFetch:
    """Test CaptionCache.get_or_fetch convenience method"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory"""
        cache_dir = tmp_path / "test_caption_cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def mock_config(self, temp_cache_dir):
        """Create a mock CaptionFirstConfig"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 30
        config.cache_captions = True
        return config

    @pytest.fixture
    def cache(self, mock_config):
        """Create a CaptionCache instance"""
        return CaptionCache(mock_config)

    def test_get_or_fetch_cache_hit(self, cache):
        """Test get_or_fetch returns cached value when available"""
        # Pre-populate cache
        segments = [CaptionSegment(0, 0.0, 2.0, "Cached text", "dQw4w9WgXcQ")]
        cached_result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=segments,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )
        cache.store(cached_result)

        # Mock fetcher that should NOT be called
        mock_fetcher = Mock(spec=CaptionFetcher)

        # Get from cache
        result = cache.get_or_fetch(mock_fetcher, "dQw4w9WgXcQ", "en")

        # Fetcher should not be called
        mock_fetcher.fetch_captions.assert_not_called()

        # Result should be from cache
        assert result.video_id == "dQw4w9WgXcQ"
        assert result.segments[0].text == "Cached text"

    def test_get_or_fetch_cache_miss(self, cache):
        """Test get_or_fetch fetches and caches on miss"""
        # Mock fetcher
        fresh_segments = [CaptionSegment(0, 0.0, 2.0, "Fresh text", "dQw4w9WgXcQ")]
        fresh_result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=fresh_segments,
            language="en",
            is_auto_generated=True,
            format_source="json3"
        )
        mock_fetcher = Mock(spec=CaptionFetcher)
        mock_fetcher.fetch_captions.return_value = fresh_result

        # Get (should fetch)
        result = cache.get_or_fetch(mock_fetcher, "dQw4w9WgXcQ", "en")

        # Fetcher should be called
        mock_fetcher.fetch_captions.assert_called_once_with(
            "dQw4w9WgXcQ",
            language="en",
            prefer_manual=True
        )

        # Result should be fresh
        assert result.segments[0].text == "Fresh text"
        assert result.is_auto_generated is True

        # Should now be cached
        cached = cache.get_caption("dQw4w9WgXcQ", "en")
        assert cached is not None
        assert cached.segments[0]['text'] == "Fresh text"

    def test_get_or_fetch_propagates_error(self, cache):
        """Test get_or_fetch propagates fetcher errors"""
        mock_fetcher = Mock(spec=CaptionFetcher)
        mock_fetcher.fetch_captions.side_effect = CaptionUnavailableError(
            "dQw4w9WgXcQ", "No captions"
        )

        with pytest.raises(CaptionUnavailableError):
            cache.get_or_fetch(mock_fetcher, "dQw4w9WgXcQ", "en")


class TestCaptionCacheStats:
    """Test CaptionCache statistics"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory"""
        cache_dir = tmp_path / "test_caption_cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def mock_config(self, temp_cache_dir):
        """Create a mock CaptionFirstConfig"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 30
        config.cache_captions = True
        return config

    @pytest.fixture
    def cache(self, mock_config):
        """Create a CaptionCache instance"""
        return CaptionCache(mock_config)

    def test_stats_empty_cache(self, cache):
        """Test stats for empty cache"""
        stats = cache.get_stats()

        assert stats['total_entries'] == 0
        assert stats['total_segments'] == 0
        assert stats['auto_generated_entries'] == 0
        assert stats['manual_entries'] == 0
        assert stats['enabled'] is True
        assert stats['max_age_days'] == 30

    def test_stats_with_entries(self, cache):
        """Test stats with cached entries"""
        # Store some entries
        for i, (video_id, auto) in enumerate([
            ("dQw4w9WgXcQ", False),
            ("abc123def45", True),
            ("xyz987uvw01", False),
        ]):
            segments = [
                CaptionSegment(0, 0.0, 2.0, "Segment 1", video_id),
                CaptionSegment(1, 2.0, 4.0, "Segment 2", video_id),
            ]
            result = CaptionResult(
                video_id=video_id,
                segments=segments,
                language="en",
                is_auto_generated=auto,
                format_source="vtt"
            )
            cache.store(result)

        stats = cache.get_stats()

        assert stats['total_entries'] == 3
        assert stats['total_segments'] == 6  # 3 videos * 2 segments each
        assert stats['auto_generated_entries'] == 1
        assert stats['manual_entries'] == 2

    def test_stats_hit_miss_tracking(self, cache):
        """Test hit/miss statistics"""
        # Store an entry
        segments = [CaptionSegment(0, 0.0, 2.0, "Test", "dQw4w9WgXcQ")]
        result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=segments,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )
        cache.store(result)

        # Generate some hits and misses
        cache.get_caption("dQw4w9WgXcQ", "en")  # Hit
        cache.get_caption("dQw4w9WgXcQ", "en")  # Hit
        cache.get_caption("nonexistent1", "en")  # Miss
        cache.get_caption("dQw4w9WgXcQ", "es")  # Miss

        stats = cache.get_stats()

        assert stats['hits'] == 2
        assert stats['misses'] == 2
        assert stats['hit_rate'] == 0.5


class TestCaptionCachePersistence:
    """Test CaptionCache persistence across instances"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory"""
        cache_dir = tmp_path / "test_caption_cache"
        cache_dir.mkdir()
        return cache_dir

    def test_persistence_across_instances(self, temp_cache_dir):
        """Test cached data persists across cache instances"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 30
        config.cache_captions = True

        # Create first cache instance and store
        cache1 = CaptionCache(config)
        segments = [CaptionSegment(0, 0.0, 2.0, "Persistent text", "dQw4w9WgXcQ")]
        result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=segments,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )
        cache1.store(result)

        # Create second cache instance and retrieve
        cache2 = CaptionCache(config)
        cached = cache2.get_caption("dQw4w9WgXcQ", "en")

        assert cached is not None
        assert cached.segments[0]['text'] == "Persistent text"

    def test_index_file_created(self, temp_cache_dir):
        """Test index file is created"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 30
        config.cache_captions = True

        cache = CaptionCache(config)

        # Store something
        segments = [CaptionSegment(0, 0.0, 2.0, "Test", "dQw4w9WgXcQ")]
        result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=segments,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )
        cache.store(result)

        # Check index file exists
        index_path = temp_cache_dir / "caption_cache_index.json"
        assert index_path.exists()

        # Verify index content
        with open(index_path) as f:
            index_data = json.load(f)

        assert "dQw4w9WgXcQ_en" in index_data


class TestCaptionCacheConfigIntegration:
    """Test CaptionCache with real CaptionFirstConfig"""

    def test_with_real_config_dataclass(self, tmp_path):
        """Test cache with actual CaptionFirstConfig dataclass"""
        from src.config.sections.download import CaptionFirstConfig

        config = CaptionFirstConfig(
            enabled=True,
            cache_captions=True,
            cache_dir=str(tmp_path / "caption_cache"),
            max_cache_age_days=7
        )

        cache = CaptionCache(config)

        assert cache.enabled is True
        assert cache.max_age_days == 7

        # Store and retrieve
        segments = [CaptionSegment(0, 0.0, 2.0, "Test", "dQw4w9WgXcQ")]
        result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=segments,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )
        cache.store(result)

        cached = cache.get_caption("dQw4w9WgXcQ", "en")
        assert cached is not None
        assert cached.segments[0]['text'] == "Test"
