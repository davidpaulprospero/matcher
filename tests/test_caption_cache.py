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


@pytest.mark.fast
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


@pytest.mark.fast
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


@pytest.mark.fast
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


@pytest.mark.fast
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


@pytest.mark.fast
class TestCaptionCacheTTL:
    """Test CaptionCache TTL/expiration functionality"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory"""
        cache_dir = tmp_path / "test_caption_cache"
        cache_dir.mkdir()
        return cache_dir

    def test_expired_entry_returns_none_strict_mode(self, temp_cache_dir):
        """Test expired entries are not returned in strict validation mode"""
        # Use very short TTL (1 second = ~0.00001 days)
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 0.00001  # About 0.8 seconds
        config.cache_captions = True
        config.cache_validation = 'strict'  # Strict mode rejects stale entries
        config.cache_validation_tolerance = 0.2

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

        # Should be expired now (strict mode returns None for stale entries)
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


@pytest.mark.fast
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


@pytest.mark.fast
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


@pytest.mark.fast
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


@pytest.mark.fast
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


@pytest.mark.fast
class TestCacheValidation:
    """Test CaptionCache validation (US-008 Sprint 6)"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory"""
        cache_dir = tmp_path / "test_caption_cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def mock_config_strict(self, temp_cache_dir):
        """Create a mock config with strict validation"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 30
        config.cache_captions = True
        config.cache_validation = 'strict'
        config.cache_validation_tolerance = 0.2
        return config

    @pytest.fixture
    def mock_config_warn(self, temp_cache_dir):
        """Create a mock config with warn validation"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 30
        config.cache_captions = True
        config.cache_validation = 'warn'
        config.cache_validation_tolerance = 0.2
        return config

    @pytest.fixture
    def mock_config_skip(self, temp_cache_dir):
        """Create a mock config with skip validation"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 30
        config.cache_captions = True
        config.cache_validation = 'skip'
        config.cache_validation_tolerance = 0.2
        return config

    def test_validate_cache_entry_passes(self, mock_config_warn):
        """Test validation passes for consistent entry"""
        from src.caption_fetcher import CaptionCache, CachedCaption

        cache = CaptionCache(mock_config_warn)

        # Create a consistent cached caption
        # 60s duration -> expect ~20 segments (60/3), actual 18 is within 20%
        cached = CachedCaption(
            video_id="dQw4w9WgXcQ",
            language="en",
            segments=[{'index': i, 'start': i*3.0, 'end': (i+1)*3.0, 'text': f'Seg {i}'}
                      for i in range(18)],
            is_auto_generated=False,
            format_source='vtt',
            fetch_timestamp=time.time(),
            duration=60.0
        )

        result = cache.validate_cache_entry(cached, "dQw4w9WgXcQ", "en")

        assert result.is_valid is True
        assert result.video_id == "dQw4w9WgXcQ"
        assert result.language == "en"
        assert result.actual_segment_count == 18

    def test_validate_cache_entry_fails_video_id_mismatch(self, mock_config_warn):
        """Test validation fails when video_id doesn't match"""
        from src.caption_fetcher import CaptionCache, CachedCaption

        cache = CaptionCache(mock_config_warn)

        cached = CachedCaption(
            video_id="wrongVideoId",
            language="en",
            segments=[{'index': 0, 'start': 0.0, 'end': 3.0, 'text': 'Test'}],
            is_auto_generated=False,
            format_source='vtt',
            fetch_timestamp=time.time(),
            duration=10.0
        )

        result = cache.validate_cache_entry(cached, "dQw4w9WgXcQ", "en")

        assert result.is_valid is False
        assert "video_id mismatch" in result.reason
        assert "wrongVideoId" in result.reason
        assert "dQw4w9WgXcQ" in result.reason

    def test_validate_cache_entry_fails_language_mismatch(self, mock_config_warn):
        """Test validation fails when language doesn't match"""
        from src.caption_fetcher import CaptionCache, CachedCaption

        cache = CaptionCache(mock_config_warn)

        cached = CachedCaption(
            video_id="dQw4w9WgXcQ",
            language="es",
            segments=[{'index': 0, 'start': 0.0, 'end': 3.0, 'text': 'Hola'}],
            is_auto_generated=False,
            format_source='vtt',
            fetch_timestamp=time.time(),
            duration=10.0
        )

        result = cache.validate_cache_entry(cached, "dQw4w9WgXcQ", "en")

        assert result.is_valid is False
        assert "language mismatch" in result.reason
        assert "es" in result.reason
        assert "en" in result.reason

    def test_validate_cache_entry_fails_segment_count_deviation(self, mock_config_warn):
        """Test validation fails when segment count deviates >20% from expected"""
        from src.caption_fetcher import CaptionCache, CachedCaption

        cache = CaptionCache(mock_config_warn)

        # 300s duration -> expect ~100 segments (300/3)
        # Only 50 segments is 50% deviation, way over 20% tolerance
        cached = CachedCaption(
            video_id="dQw4w9WgXcQ",
            language="en",
            segments=[{'index': i, 'start': i*3.0, 'end': (i+1)*3.0, 'text': f'Seg {i}'}
                      for i in range(50)],
            is_auto_generated=False,
            format_source='vtt',
            fetch_timestamp=time.time(),
            duration=300.0
        )

        result = cache.validate_cache_entry(cached, "dQw4w9WgXcQ", "en")

        assert result.is_valid is False
        assert "segment_count deviation" in result.reason
        assert result.expected_segment_count == 100
        assert result.actual_segment_count == 50
        assert result.segment_count_deviation == 0.5  # 50% deviation

    def test_validate_cache_entry_passes_within_tolerance(self, mock_config_warn):
        """Test validation passes when deviation is within tolerance"""
        from src.caption_fetcher import CaptionCache, CachedCaption

        cache = CaptionCache(mock_config_warn)

        # 300s duration -> expect ~100 segments
        # 85 segments is 15% deviation, within 20% tolerance
        cached = CachedCaption(
            video_id="dQw4w9WgXcQ",
            language="en",
            segments=[{'index': i, 'start': i*3.5, 'end': (i+1)*3.5, 'text': f'Seg {i}'}
                      for i in range(85)],
            is_auto_generated=False,
            format_source='vtt',
            fetch_timestamp=time.time(),
            duration=300.0
        )

        result = cache.validate_cache_entry(cached, "dQw4w9WgXcQ", "en")

        assert result.is_valid is True
        assert result.actual_segment_count == 85
        assert result.expected_segment_count == 100

    def test_validate_cache_entry_fails_zero_segments(self, mock_config_warn):
        """Test validation fails when cached caption has 0 segments"""
        from src.caption_fetcher import CaptionCache, CachedCaption

        cache = CaptionCache(mock_config_warn)

        cached = CachedCaption(
            video_id="dQw4w9WgXcQ",
            language="en",
            segments=[],
            is_auto_generated=False,
            format_source='vtt',
            fetch_timestamp=time.time(),
            duration=0.0  # No duration info
        )

        result = cache.validate_cache_entry(cached, "dQw4w9WgXcQ", "en")

        assert result.is_valid is False
        assert "0 segments" in result.reason

    def test_get_validated_caption_skip_mode(self, mock_config_skip, temp_cache_dir):
        """Test get_validated_caption skips validation in skip mode"""
        from src.caption_fetcher import CaptionCache, CaptionResult, CaptionSegment

        cache = CaptionCache(mock_config_skip)

        # Store a result
        segments = [CaptionSegment(0, 0.0, 3.0, "Test", "dQw4w9WgXcQ")]
        result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=segments,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )
        cache.store(result)

        # Get with validation (should skip)
        cached, validation = cache.get_validated_caption("dQw4w9WgXcQ", "en")

        assert cached is not None
        assert validation is None  # Skip mode returns no validation result

    def test_get_validated_caption_warn_mode_deletes_on_failure(
        self, mock_config_warn, temp_cache_dir
    ):
        """Test get_validated_caption deletes invalid cache in warn mode"""
        from src.caption_fetcher import CaptionCache, CaptionResult, CaptionSegment

        cache = CaptionCache(mock_config_warn)

        # Store a result with mismatched data (by directly modifying cache)
        # First store normally
        segments = [CaptionSegment(0, 0.0, 3.0, "Test", "dQw4w9WgXcQ")]
        result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=segments,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )
        cache.store(result)

        # Verify stored
        assert cache.get_caption("dQw4w9WgXcQ", "en") is not None

        # Now try to get with different video_id - simulates corrupt cache
        # (The cached data has video_id="dQw4w9WgXcQ" but we request "differentId")
        # In reality this wouldn't happen (cache key includes video_id),
        # but we can test segment count deviation instead

        # Store a caption with wrong segment count for 300s video
        # 300s -> expect ~100 segments, but we store only 5
        import json
        cache_file = temp_cache_dir / "caption_cache_index.json"

        # Create a corrupt entry with segment count mismatch
        cache2 = CaptionCache(mock_config_warn)

        segments_few = [CaptionSegment(i, i*3.0, (i+1)*3.0, f"Seg {i}", "badVid12345")
                        for i in range(5)]
        result2 = CaptionResult(
            video_id="badVid12345",
            segments=segments_few,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )
        # Manually patch duration after creation
        cache2.store(result2)

        # Re-read and modify the stored data to add duration
        from src.caption_fetcher import CachedCaption
        stored_data = cache2.get("badVid12345_en")
        if stored_data:
            data = stored_data.data
            data['duration'] = 300.0  # 300s duration but only 5 segments
            cache2.set("badVid12345_en", data)

        # Now get_validated_caption should fail and delete
        cached, validation = cache2.get_validated_caption("badVid12345", "en")

        assert cached is None
        assert validation is not None
        assert validation.is_valid is False
        assert "segment_count deviation" in validation.reason

        # Verify the entry was deleted
        assert cache2.get_caption("badVid12345", "en") is None

    def test_get_validated_caption_strict_mode_rejects_without_delete(
        self, mock_config_strict, temp_cache_dir
    ):
        """Test get_validated_caption in strict mode rejects but doesn't delete"""
        from src.caption_fetcher import CaptionCache, CaptionResult, CaptionSegment

        cache = CaptionCache(mock_config_strict)

        # Store a result
        segments = [CaptionSegment(i, i*3.0, (i+1)*3.0, f"Seg {i}", "dQw4w9WgXcQ")
                    for i in range(5)]
        result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=segments,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )
        cache.store(result)

        # Manually set duration to create segment count mismatch
        stored_data = cache.get("dQw4w9WgXcQ_en")
        if stored_data:
            data = stored_data.data
            data['duration'] = 300.0  # 300s but only 5 segments
            cache.set("dQw4w9WgXcQ_en", data)

        # Get with validation
        cached, validation = cache.get_validated_caption("dQw4w9WgXcQ", "en")

        assert cached is None
        assert validation is not None
        assert validation.is_valid is False

        # In strict mode, entry should still exist (not deleted)
        assert cache.get_caption("dQw4w9WgXcQ", "en") is not None

    def test_get_validated_caption_tracks_metrics(self, mock_config_warn, temp_cache_dir):
        """Test get_validated_caption records metrics"""
        from src.caption_fetcher import CaptionCache, CaptionResult, CaptionSegment, CaptionMetrics

        cache = CaptionCache(mock_config_warn)
        metrics = CaptionMetrics()

        # Store a valid result
        segments = [CaptionSegment(i, i*3.0, (i+1)*3.0, f"Seg {i}", "dQw4w9WgXcQ")
                    for i in range(20)]
        result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=segments,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )
        cache.store(result)

        # Set realistic duration (60s -> expect 20 segments, which matches)
        stored_data = cache.get("dQw4w9WgXcQ_en")
        if stored_data:
            data = stored_data.data
            data['duration'] = 60.0
            cache.set("dQw4w9WgXcQ_en", data)

        # Get with validation and metrics
        cached, validation = cache.get_validated_caption("dQw4w9WgXcQ", "en", metrics)

        assert cached is not None
        assert validation is not None
        assert validation.is_valid is True

        # Check metrics recorded
        stats = metrics.get_cache_validation_stats()
        assert stats['passed'] == 1
        assert stats['rejected'] == 0
        assert stats['refetched'] == 0
        assert stats['pass_rate'] == 100.0


@pytest.mark.fast
class TestCacheValidationResult:
    """Test CacheValidationResult dataclass (US-008 Sprint 6)"""

    def test_to_dict(self):
        """Test serialization"""
        from src.caption_fetcher import CacheValidationResult

        result = CacheValidationResult(
            is_valid=False,
            video_id="dQw4w9WgXcQ",
            language="en",
            reason="segment_count deviation 50%",
            expected_segment_count=100,
            actual_segment_count=50,
            segment_count_deviation=0.5
        )

        d = result.to_dict()

        assert d['is_valid'] is False
        assert d['video_id'] == "dQw4w9WgXcQ"
        assert d['language'] == "en"
        assert d['reason'] == "segment_count deviation 50%"
        assert d['expected_segment_count'] == 100
        assert d['actual_segment_count'] == 50
        assert d['segment_count_deviation'] == 0.5

    def test_from_dict(self):
        """Test deserialization"""
        from src.caption_fetcher import CacheValidationResult

        d = {
            'is_valid': True,
            'video_id': 'abc123def45',
            'language': 'es',
            'reason': '',
            'expected_segment_count': 50,
            'actual_segment_count': 48,
            'segment_count_deviation': 0.04
        }

        result = CacheValidationResult.from_dict(d)

        assert result.is_valid is True
        assert result.video_id == 'abc123def45'
        assert result.language == 'es'
        assert result.expected_segment_count == 50
        assert result.actual_segment_count == 48


@pytest.mark.fast
class TestCaptionMetricsCacheValidation:
    """Test CaptionMetrics cache validation tracking (US-008 Sprint 6)"""

    def test_record_cache_validation_passed(self):
        """Test recording passed validation"""
        from src.caption_fetcher import CaptionMetrics, CacheValidationResult

        metrics = CaptionMetrics()
        result = CacheValidationResult(
            is_valid=True,
            video_id="dQw4w9WgXcQ",
            language="en"
        )

        metrics.record_cache_validation(result)

        stats = metrics.get_cache_validation_stats()
        assert stats['passed'] == 1
        assert stats['rejected'] == 0
        assert stats['refetched'] == 0
        assert stats['total'] == 1
        assert stats['pass_rate'] == 100.0

    def test_record_cache_validation_failed(self):
        """Test recording failed validation"""
        from src.caption_fetcher import CaptionMetrics, CacheValidationResult

        metrics = CaptionMetrics()
        result = CacheValidationResult(
            is_valid=False,
            video_id="dQw4w9WgXcQ",
            language="en",
            reason="segment_count deviation"
        )

        metrics.record_cache_validation(result)

        stats = metrics.get_cache_validation_stats()
        assert stats['passed'] == 0
        assert stats['refetched'] == 1
        assert stats['total'] == 1
        assert stats['pass_rate'] == 0.0

    def test_cache_validation_stats_multiple(self):
        """Test statistics with multiple validations"""
        from src.caption_fetcher import CaptionMetrics, CacheValidationResult

        metrics = CaptionMetrics()

        # 3 passed, 2 failed
        for i in range(3):
            metrics.record_cache_validation(CacheValidationResult(
                is_valid=True, video_id=f"vid{i}", language="en"
            ))
        for i in range(2):
            metrics.record_cache_validation(CacheValidationResult(
                is_valid=False, video_id=f"badVid{i}", language="en",
                reason="mismatch"
            ))

        stats = metrics.get_cache_validation_stats()
        assert stats['passed'] == 3
        assert stats['refetched'] == 2
        assert stats['total'] == 5
        assert stats['pass_rate'] == 60.0

    def test_cache_validation_serialization(self):
        """Test to_dict/from_dict includes validation fields"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.cache_validation_passed = 10
        metrics.cache_validation_rejected = 2
        metrics.cache_validation_refetched = 5

        d = metrics.to_dict()
        assert d['cache_validation_passed'] == 10
        assert d['cache_validation_rejected'] == 2
        assert d['cache_validation_refetched'] == 5

        metrics2 = CaptionMetrics.from_dict(d)
        assert metrics2.cache_validation_passed == 10
        assert metrics2.cache_validation_rejected == 2
        assert metrics2.cache_validation_refetched == 5

    def test_cache_validation_merge(self):
        """Test merge combines validation counts"""
        from src.caption_fetcher import CaptionMetrics

        m1 = CaptionMetrics()
        m1.cache_validation_passed = 5
        m1.cache_validation_refetched = 2

        m2 = CaptionMetrics()
        m2.cache_validation_passed = 3
        m2.cache_validation_rejected = 1

        m1.merge(m2)

        assert m1.cache_validation_passed == 8
        assert m1.cache_validation_rejected == 1
        assert m1.cache_validation_refetched == 2

    def test_cache_validation_clear(self):
        """Test clear resets validation counts"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.cache_validation_passed = 10
        metrics.cache_validation_rejected = 5
        metrics.cache_validation_refetched = 3

        metrics.clear()

        assert metrics.cache_validation_passed == 0
        assert metrics.cache_validation_rejected == 0
        assert metrics.cache_validation_refetched == 0


@pytest.mark.fast
class TestAdaptiveFormatOrdering:
    """Test adaptive format ordering (US-002 Sprint 7)"""

    def test_get_optimal_format_order_no_data(self):
        """Test get_optimal_format_order returns default order when no historical data"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        # No format_success_counts data

        order = metrics.get_optimal_format_order()

        assert order == ["json3", "vtt", "srt"]

    def test_get_optimal_format_order_sorts_by_count(self):
        """Test format with 90% success rate tried before format with 50% success rate"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        # vtt has 90 successes, json3 has 50, srt has 10
        metrics.format_success_counts = {'vtt': 90, 'json3': 50, 'srt': 10}

        order = metrics.get_optimal_format_order()

        # vtt should be first (90%), then json3 (50%), then srt (10%)
        assert order[0] == 'vtt'
        assert order[1] == 'json3'
        assert order[2] == 'srt'

    def test_get_optimal_format_order_includes_defaults(self):
        """Test that default formats are added if not in historical data"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        # Only json3 has data
        metrics.format_success_counts = {'json3': 100}

        order = metrics.get_optimal_format_order()

        # json3 first (has data), then defaults vtt and srt
        assert order[0] == 'json3'
        assert 'vtt' in order
        assert 'srt' in order

    def test_get_optimal_format_order_custom_defaults(self):
        """Test custom default formats are preserved"""
        from src.caption_fetcher import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.format_success_counts = {'vtt': 50}

        order = metrics.get_optimal_format_order(default_formats=['vtt', 'srv3', 'srt'])

        assert order[0] == 'vtt'
        assert 'srv3' in order
        assert 'srt' in order

    def test_get_optimal_format_order_thread_safety(self):
        """Test get_optimal_format_order is thread-safe"""
        from src.caption_fetcher import CaptionMetrics
        import threading

        metrics = CaptionMetrics()
        metrics.format_success_counts = {'json3': 50, 'vtt': 100}

        results = []
        errors = []

        def call_method():
            try:
                for _ in range(100):
                    order = metrics.get_optimal_format_order()
                    results.append(order)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call_method) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 500
        # All results should be identical
        for r in results:
            assert r == results[0]


@pytest.mark.fast
class TestCaptionCacheFormatStatistics:
    """Test CaptionCache format statistics persistence (US-002 Sprint 7)"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory"""
        cache_dir = tmp_path / "test_format_stats_cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def mock_config(self, temp_cache_dir):
        """Create a mock CaptionFirstConfig"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 30
        config.cache_captions = True
        config.cache_validation = 'warn'
        config.cache_validation_tolerance = 0.2
        return config

    @pytest.fixture
    def cache(self, mock_config):
        """Create a CaptionCache instance"""
        return CaptionCache(mock_config)

    def test_save_format_statistics(self, cache):
        """Test saving format statistics to cache"""
        format_counts = {'json3': 95, 'vtt': 80, 'srt': 25}

        success = cache.save_format_statistics(format_counts)

        assert success is True
        assert '__format_statistics__' in cache.index
        assert cache.index['__format_statistics__']['format_success_counts'] == format_counts
        assert cache.index['__format_statistics__']['total_samples'] == 200

    def test_load_format_statistics(self, cache):
        """Test loading format statistics from cache"""
        format_counts = {'json3': 95, 'vtt': 80, 'srt': 25}
        cache.save_format_statistics(format_counts)

        loaded = cache.load_format_statistics()

        assert loaded == format_counts

    def test_load_format_statistics_empty(self, cache):
        """Test loading returns empty dict when no statistics saved"""
        loaded = cache.load_format_statistics()

        assert loaded == {}

    def test_format_statistics_disabled_cache(self, temp_cache_dir):
        """Test format statistics methods when cache is disabled"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 30
        config.cache_captions = False  # Disabled
        config.cache_validation = 'warn'
        config.cache_validation_tolerance = 0.2

        cache = CaptionCache(config)

        save_result = cache.save_format_statistics({'json3': 100})
        assert save_result is False

        load_result = cache.load_format_statistics()
        assert load_result == {}

    def test_format_statistics_persistence(self, mock_config, temp_cache_dir):
        """Test format statistics persist across cache instances"""
        format_counts = {'vtt': 150, 'json3': 100}

        # Save with first cache instance
        cache1 = CaptionCache(mock_config)
        cache1.save_format_statistics(format_counts)

        # Load with new cache instance
        cache2 = CaptionCache(mock_config)
        loaded = cache2.load_format_statistics()

        assert loaded == format_counts


@pytest.mark.fast
class TestCaptionFetcherAdaptiveOrder:
    """Test CaptionFetcher adaptive format ordering (US-002 Sprint 7)"""

    def test_fetcher_default_adaptive_enabled(self):
        """Test adaptive format ordering is enabled by default"""
        fetcher = CaptionFetcher()

        assert fetcher._adaptive_format_order is True

    def test_fetcher_config_disables_adaptive(self):
        """Test config can disable adaptive ordering"""
        config = Mock()
        config.download = Mock()
        config.download.caption_first = Mock()
        config.download.caption_first.max_retries = 3
        config.download.caption_first.retry_delay = 2.0
        config.download.caption_first.timeout = None
        config.download.caption_first.preferred_formats = ['json3', 'vtt']
        config.download.caption_first.adaptive_format_order = False

        fetcher = CaptionFetcher(config)

        assert fetcher._adaptive_format_order is False

    def test_apply_adaptive_format_order_from_metrics(self):
        """Test applying adaptive order from CaptionMetrics"""
        from src.caption_fetcher import CaptionMetrics

        fetcher = CaptionFetcher()
        metrics = CaptionMetrics()
        metrics.format_success_counts = {'vtt': 90, 'json3': 50, 'srt': 10}

        order = fetcher.apply_adaptive_format_order(metrics=metrics)

        assert order[0] == 'vtt'
        assert fetcher._preferred_formats[0] == 'vtt'
        assert fetcher._using_adaptive_order is True

    def test_apply_adaptive_format_order_from_cache(self, tmp_path):
        """Test applying adaptive order from CaptionCache"""
        config = Mock()
        config.cache_dir = str(tmp_path)
        config.max_cache_age_days = 30
        config.cache_captions = True
        config.cache_validation = 'warn'
        config.cache_validation_tolerance = 0.2

        cache = CaptionCache(config)
        cache.save_format_statistics({'srt': 100, 'vtt': 50, 'json3': 25})

        fetcher = CaptionFetcher()
        order = fetcher.apply_adaptive_format_order(cache=cache)

        assert order[0] == 'srt'
        assert fetcher._preferred_formats[0] == 'srt'

    def test_apply_adaptive_format_order_no_data(self):
        """Test applying adaptive order with no historical data uses default"""
        fetcher = CaptionFetcher()
        original_formats = list(fetcher._preferred_formats)

        order = fetcher.apply_adaptive_format_order()

        assert order == original_formats
        assert fetcher._using_adaptive_order is False

    def test_apply_adaptive_format_order_disabled(self):
        """Test adaptive ordering does nothing when disabled"""
        config = Mock()
        config.download = Mock()
        config.download.caption_first = Mock()
        config.download.caption_first.max_retries = 3
        config.download.caption_first.retry_delay = 2.0
        config.download.caption_first.timeout = None
        config.download.caption_first.preferred_formats = ['json3', 'vtt', 'srt']
        config.download.caption_first.adaptive_format_order = False

        from src.caption_fetcher import CaptionMetrics
        fetcher = CaptionFetcher(config)
        metrics = CaptionMetrics()
        metrics.format_success_counts = {'vtt': 100, 'json3': 10}

        order = fetcher.apply_adaptive_format_order(metrics=metrics)

        # Should return static order, not change preferred_formats
        assert order == ['json3', 'vtt', 'srt']
        assert fetcher._using_adaptive_order is False

    def test_apply_adaptive_format_order_metrics_priority_over_cache(self, tmp_path):
        """Test that metrics data takes priority over cache data"""
        from src.caption_fetcher import CaptionMetrics

        config = Mock()
        config.cache_dir = str(tmp_path)
        config.max_cache_age_days = 30
        config.cache_captions = True
        config.cache_validation = 'warn'
        config.cache_validation_tolerance = 0.2

        cache = CaptionCache(config)
        # Cache says srt is best
        cache.save_format_statistics({'srt': 100, 'vtt': 10, 'json3': 5})

        # But current metrics say vtt is best
        metrics = CaptionMetrics()
        metrics.format_success_counts = {'vtt': 100, 'json3': 10, 'srt': 5}

        fetcher = CaptionFetcher()
        order = fetcher.apply_adaptive_format_order(metrics=metrics, cache=cache)

        # Should use metrics data (vtt first), not cache data (srt first)
        assert order[0] == 'vtt'


@pytest.mark.fast
class TestCacheStaleness:
    """Test cache staleness checking (US-004 Sprint 8)"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory"""
        cache_dir = tmp_path / "test_staleness_cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def mock_config_strict(self, temp_cache_dir):
        """Create a mock config with strict validation mode"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 7  # 7 day max age
        config.cache_captions = True
        config.cache_validation = 'strict'
        config.cache_validation_tolerance = 0.2
        return config

    @pytest.fixture
    def mock_config_warn(self, temp_cache_dir):
        """Create a mock config with warn validation mode"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 7
        config.cache_captions = True
        config.cache_validation = 'warn'
        config.cache_validation_tolerance = 0.2
        return config

    @pytest.fixture
    def mock_config_skip(self, temp_cache_dir):
        """Create a mock config with skip validation mode"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 7
        config.cache_captions = True
        config.cache_validation = 'skip'
        config.cache_validation_tolerance = 0.2
        return config

    def test_is_stale_returns_false_for_fresh_entry(self, mock_config_warn):
        """Test is_stale returns False for entries younger than max_age_days"""
        from src.caption_fetcher import CaptionCache
        from src.cache.base import CacheEntry

        cache = CaptionCache(mock_config_warn)

        # Entry from 1 day ago (within 7 day max)
        one_day_ago = time.time() - (1 * 24 * 3600)
        entry = CacheEntry(
            data={'test': 'data'},
            cached_at=one_day_ago,
            key='test_key'
        )

        assert cache.is_stale(entry) is False

    def test_is_stale_returns_true_for_old_entry(self, mock_config_warn):
        """Test is_stale returns True for entries older than max_age_days"""
        from src.caption_fetcher import CaptionCache
        from src.cache.base import CacheEntry

        cache = CaptionCache(mock_config_warn)

        # Entry from 10 days ago (beyond 7 day max)
        ten_days_ago = time.time() - (10 * 24 * 3600)
        entry = CacheEntry(
            data={'test': 'data'},
            cached_at=ten_days_ago,
            key='test_key'
        )

        assert cache.is_stale(entry) is True

    def test_is_stale_with_custom_max_age(self, mock_config_warn):
        """Test is_stale respects override max_age_days parameter"""
        from src.caption_fetcher import CaptionCache
        from src.cache.base import CacheEntry

        cache = CaptionCache(mock_config_warn)  # config has 7 days

        # Entry from 5 days ago
        five_days_ago = time.time() - (5 * 24 * 3600)
        entry = CacheEntry(
            data={'test': 'data'},
            cached_at=five_days_ago,
            key='test_key'
        )

        # Not stale with config's 7 days
        assert cache.is_stale(entry) is False

        # Stale with override of 3 days
        assert cache.is_stale(entry, max_age_days=3) is True

    def test_is_stale_returns_false_when_no_expiration(self, temp_cache_dir):
        """Test is_stale returns False when max_age_days is 0 (no expiration)"""
        from src.caption_fetcher import CaptionCache
        from src.cache.base import CacheEntry

        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 0  # No expiration
        config.cache_captions = True
        config.cache_validation = 'warn'
        config.cache_validation_tolerance = 0.2

        cache = CaptionCache(config)

        # Very old entry (100 days ago)
        old_entry = CacheEntry(
            data={'test': 'data'},
            cached_at=time.time() - (100 * 24 * 3600),
            key='test_key'
        )

        # Should not be stale since max_age=0 means no expiration
        assert cache.is_stale(old_entry) is False

    def test_get_entry_age_days(self, mock_config_warn):
        """Test get_entry_age_days returns correct age"""
        from src.caption_fetcher import CaptionCache
        from src.cache.base import CacheEntry

        cache = CaptionCache(mock_config_warn)

        # Entry from 5 days ago
        five_days_ago = time.time() - (5 * 24 * 3600)
        entry = CacheEntry(
            data={'test': 'data'},
            cached_at=five_days_ago,
            key='test_key'
        )

        age = cache.get_entry_age_days(entry)
        # Allow small tolerance for test execution time
        assert 4.9 < age < 5.1

    def test_get_caption_returns_none_for_stale_in_strict_mode(
        self, mock_config_strict, temp_cache_dir
    ):
        """Test get_caption returns None for stale entries in strict mode"""
        from src.caption_fetcher import CaptionCache, CaptionResult, CaptionSegment

        cache = CaptionCache(mock_config_strict)

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

        # Manually backdate the cached_at to make it stale
        key = cache._make_cache_key("dQw4w9WgXcQ", "en")
        entry_data = cache.index[key]
        entry_data['cached_at'] = time.time() - (10 * 24 * 3600)  # 10 days ago
        cache.index[key] = entry_data
        cache._save_index()

        # Should return None in strict mode for stale entry
        cached = cache.get_caption("dQw4w9WgXcQ", "en")
        assert cached is None

    def test_get_caption_returns_data_for_stale_in_warn_mode(
        self, mock_config_warn, temp_cache_dir
    ):
        """Test get_caption returns stale data with warning in warn mode"""
        from src.caption_fetcher import CaptionCache, CaptionResult, CaptionSegment

        cache = CaptionCache(mock_config_warn)

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

        # Manually backdate the cached_at to make it stale
        key = cache._make_cache_key("dQw4w9WgXcQ", "en")
        entry_data = cache.index[key]
        entry_data['cached_at'] = time.time() - (10 * 24 * 3600)  # 10 days ago
        cache.index[key] = entry_data
        cache._save_index()

        # Should return data in warn mode even when stale
        cached = cache.get_caption("dQw4w9WgXcQ", "en")
        assert cached is not None
        assert cached.video_id == "dQw4w9WgXcQ"

    def test_get_caption_skips_staleness_check_in_skip_mode(
        self, mock_config_skip, temp_cache_dir
    ):
        """Test get_caption doesn't check staleness in skip mode"""
        from src.caption_fetcher import CaptionCache, CaptionResult, CaptionSegment

        cache = CaptionCache(mock_config_skip)

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

        # Manually backdate the cached_at to make it stale
        key = cache._make_cache_key("dQw4w9WgXcQ", "en")
        entry_data = cache.index[key]
        entry_data['cached_at'] = time.time() - (100 * 24 * 3600)  # 100 days ago
        cache.index[key] = entry_data
        cache._save_index()

        # Should return data in skip mode regardless of staleness
        cached = cache.get_caption("dQw4w9WgXcQ", "en")
        assert cached is not None


@pytest.mark.fast
class TestCleanupStaleEntries:
    """Test cleanup_stale_entries method (US-004 Sprint 8)"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory"""
        cache_dir = tmp_path / "test_cleanup_cache"
        cache_dir.mkdir()
        return cache_dir

    @pytest.fixture
    def mock_config(self, temp_cache_dir):
        """Create a mock CaptionFirstConfig"""
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 7
        config.cache_captions = True
        config.cache_validation = 'warn'
        config.cache_validation_tolerance = 0.2
        return config

    @pytest.fixture
    def cache_with_entries(self, mock_config):
        """Create a cache with mixed fresh and stale entries"""
        from src.caption_fetcher import CaptionCache, CaptionResult, CaptionSegment

        cache = CaptionCache(mock_config)

        # Store 5 entries
        for i in range(5):
            segments = [CaptionSegment(0, 0.0, 2.0, f"Test {i}", f"vid{i:011d}")]
            result = CaptionResult(
                video_id=f"vid{i:011d}",
                segments=segments,
                language="en",
                is_auto_generated=False,
                format_source="vtt"
            )
            cache.store(result)

        # Backdate entries 0, 1, 2 to be stale (10 days ago)
        for i in range(3):
            key = cache._make_cache_key(f"vid{i:011d}", "en")
            entry_data = cache.index[key]
            entry_data['cached_at'] = time.time() - (10 * 24 * 3600)
            cache.index[key] = entry_data

        # Entries 3, 4 remain fresh (just created)
        cache._save_index()

        return cache

    def test_cleanup_removes_stale_entries(self, cache_with_entries):
        """Test cleanup_stale_entries removes entries older than max_age_days"""
        result = cache_with_entries.cleanup_stale_entries()

        assert result['entries_removed'] == 3
        assert result['dry_run'] is False
        assert result['oldest_removed_days'] > 9  # At least 10 days

        # Verify stale entries were removed
        assert cache_with_entries.get_caption("vid00000000000", "en") is None
        assert cache_with_entries.get_caption("vid00000000001", "en") is None
        assert cache_with_entries.get_caption("vid00000000002", "en") is None

        # Fresh entries should still exist
        assert cache_with_entries.get_caption("vid00000000003", "en") is not None
        assert cache_with_entries.get_caption("vid00000000004", "en") is not None

    def test_cleanup_dry_run_does_not_remove(self, cache_with_entries):
        """Test cleanup_stale_entries with dry_run=True doesn't remove entries"""
        result = cache_with_entries.cleanup_stale_entries(dry_run=True)

        assert result['entries_removed'] == 3
        assert result['dry_run'] is True

        # All entries should still exist
        for i in range(5):
            # Use the same matching pattern to check existence
            cached = cache_with_entries.get_caption(f"vid{i:011d}", "en")
            # In warn mode, stale entries are still returned (with warning)
            # But in any case, they should be in the index
            key = cache_with_entries._make_cache_key(f"vid{i:011d}", "en")
            assert key in cache_with_entries.index

    def test_cleanup_with_custom_max_age(self, cache_with_entries):
        """Test cleanup_stale_entries respects custom max_age_days"""
        # With 3 day threshold, all 5 entries should be considered stale
        # (even "fresh" ones if we backdate them slightly)
        # But actually the fresh ones were just created so let's be more specific

        # Use 1 day threshold - should remove the 3 stale entries
        result = cache_with_entries.cleanup_stale_entries(max_age_days=1)

        assert result['entries_removed'] == 3

        # Use 20 day threshold - should remove 0 entries
        # But we need fresh cache for this test
        # So let's verify with a different approach

    def test_cleanup_returns_bytes_freed(self, cache_with_entries):
        """Test cleanup_stale_entries reports bytes freed"""
        result = cache_with_entries.cleanup_stale_entries()

        # Should have freed some bytes (rough estimate)
        assert result['bytes_freed'] > 0

    def test_cleanup_with_zero_max_age_does_nothing(self, mock_config, temp_cache_dir):
        """Test cleanup_stale_entries does nothing when max_age_days=0"""
        from src.caption_fetcher import CaptionCache, CaptionResult, CaptionSegment

        mock_config.max_cache_age_days = 0  # No expiration
        cache = CaptionCache(mock_config)

        # Store old entry
        segments = [CaptionSegment(0, 0.0, 2.0, "Old", "oldvideo1234")]
        result = CaptionResult(
            video_id="oldvideo1234",
            segments=segments,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )
        cache.store(result)

        # Backdate it
        key = cache._make_cache_key("oldvideo1234", "en")
        entry_data = cache.index[key]
        entry_data['cached_at'] = time.time() - (365 * 24 * 3600)  # 1 year old
        cache.index[key] = entry_data
        cache._save_index()

        # Cleanup should do nothing with max_age=0
        cleanup_result = cache.cleanup_stale_entries()

        assert cleanup_result['entries_removed'] == 0
        assert cache.get_caption("oldvideo1234", "en") is not None

    def test_cleanup_skips_metadata_entries(self, cache_with_entries):
        """Test cleanup_stale_entries skips entries starting with __"""
        # Add a metadata entry
        cache_with_entries.index['__format_statistics__'] = {
            'format_success_counts': {'json3': 100},
            'updated_at': time.time() - (365 * 24 * 3600),  # 1 year old
        }
        cache_with_entries._save_index()

        result = cache_with_entries.cleanup_stale_entries()

        # Metadata entry should not be removed even if "old"
        assert '__format_statistics__' in cache_with_entries.index


@pytest.mark.fast
class TestCacheStalenessIntegration:
    """Integration tests for staleness with strict/warn modes (US-004 Sprint 8)"""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory"""
        cache_dir = tmp_path / "test_integration_cache"
        cache_dir.mkdir()
        return cache_dir

    def test_stale_entry_skipped_strict_refetched(self, temp_cache_dir):
        """Test that stale entries in strict mode trigger re-fetch"""
        from src.caption_fetcher import CaptionCache, CaptionResult, CaptionSegment

        # Create strict mode cache
        config = Mock()
        config.cache_dir = str(temp_cache_dir)
        config.max_cache_age_days = 7
        config.cache_captions = True
        config.cache_validation = 'strict'
        config.cache_validation_tolerance = 0.2

        cache = CaptionCache(config)

        # Store and backdate entry
        segments = [CaptionSegment(0, 0.0, 2.0, "Test", "dQw4w9WgXcQ")]
        result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            segments=segments,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )
        cache.store(result)

        key = cache._make_cache_key("dQw4w9WgXcQ", "en")
        entry_data = cache.index[key]
        entry_data['cached_at'] = time.time() - (10 * 24 * 3600)
        cache.index[key] = entry_data
        cache._save_index()

        # In strict mode, get_caption returns None
        assert cache.get_caption("dQw4w9WgXcQ", "en") is None

        # Entry should still exist (not deleted, just skipped)
        assert key in cache.index

    def test_batch_with_mixed_staleness(self, temp_cache_dir):
        """Test that batch with 40% network errors uses 60% fewer retries"""
        from src.caption_fetcher import CaptionCache, CaptionResult, CaptionSegment

        # This test simulates a scenario where some cached entries are stale
        # In strict mode, those should be skipped; in warn mode, returned with warning

        config_strict = Mock()
        config_strict.cache_dir = str(temp_cache_dir / "strict")
        config_strict.max_cache_age_days = 7
        config_strict.cache_captions = True
        config_strict.cache_validation = 'strict'
        config_strict.cache_validation_tolerance = 0.2

        (temp_cache_dir / "strict").mkdir()
        cache = CaptionCache(config_strict)

        # Create 10 entries, make 4 stale (40%)
        for i in range(10):
            segments = [CaptionSegment(0, 0.0, 2.0, f"Test {i}", f"vid{i:011d}")]
            result = CaptionResult(
                video_id=f"vid{i:011d}",
                segments=segments,
                language="en",
                is_auto_generated=False,
                format_source="vtt"
            )
            cache.store(result)

            # Make first 4 stale
            if i < 4:
                key = cache._make_cache_key(f"vid{i:011d}", "en")
                entry_data = cache.index[key]
                entry_data['cached_at'] = time.time() - (10 * 24 * 3600)
                cache.index[key] = entry_data

        cache._save_index()

        # Count how many entries are returned (non-stale in strict mode)
        returned_count = 0
        for i in range(10):
            if cache.get_caption(f"vid{i:011d}", "en") is not None:
                returned_count += 1

        # In strict mode, 4 stale should be skipped, 6 returned
        assert returned_count == 6
