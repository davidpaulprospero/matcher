"""
Tests for enhanced caption caching functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Import the module under test
from src.caption_fetcher_cache import EnhancedCaptionCache, CacheMetrics


class TestCacheMetrics:
    """Test CacheMetrics tracking."""
    
    def test_initial_state(self):
        """Test initial metrics state."""
        metrics = CacheMetrics()
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.hit_rate == 0.0
    
    def test_record_hit(self):
        """Test recording cache hits."""
        metrics = CacheMetrics()
        metrics.record_hit()
        assert metrics.hits == 1
        assert metrics.misses == 0
        assert metrics.hit_rate == 1.0
    
    def test_record_miss(self):
        """Test recording cache misses."""
        metrics = CacheMetrics()
        metrics.record_miss()
        assert metrics.hits == 0
        assert metrics.misses == 1
        assert metrics.hit_rate == 0.0
    
    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        metrics = CacheMetrics()
        metrics.record_hit()
        metrics.record_hit()
        metrics.record_miss()
        assert metrics.hit_rate == 2/3
    
    def test_record_warm_hit(self):
        """Test recording warm cache hits."""
        metrics = CacheMetrics()
        metrics.record_hit(is_warm=True)
        assert metrics.warm_hits == 1
    
    def test_record_bulk_hit(self):
        """Test recording bulk cache hits."""
        metrics = CacheMetrics()
        metrics.record_hit(is_bulk=True)
        assert metrics.bulk_hits == 1
    
    def test_compression_tracking(self):
        """Test compression statistics tracking."""
        metrics = CacheMetrics()
        metrics.record_compression(1000, 400)  # 60% compression
        assert metrics.compressed_entries == 1
        assert metrics.bytes_uncompressed == 1000
        assert metrics.bytes_compressed == 400
        assert metrics.compression_ratio == 0.4
    
    def test_prefetch_tracking(self):
        """Test prefetch statistics tracking."""
        metrics = CacheMetrics()
        metrics.record_prefetch(hit=True)
        metrics.record_prefetch(hit=False)
        assert metrics.prefetch_attempts == 2
        assert metrics.prefetch_hits == 1
        assert metrics.prefetch_success_rate == 0.5
    
    def test_lookup_time_tracking(self):
        """Test average lookup time tracking."""
        metrics = CacheMetrics()
        metrics.record_hit(lookup_time_ms=10.0)
        metrics.record_hit(lookup_time_ms=20.0)
        assert metrics.avg_lookup_time_ms == 15.0
    
    def test_to_dict(self):
        """Test metrics serialization."""
        metrics = CacheMetrics()
        metrics.record_hit()
        metrics.record_compression(1000, 500)
        
        data = metrics.to_dict()
        assert data['hits'] == 1
        assert data['hit_rate'] == 1.0
        assert data['compression_ratio'] == 0.5
        assert data['bytes_saved'] == 500


class TestEnhancedCaptionCache:
    """Test EnhancedCaptionCache functionality."""
    
    @pytest.fixture
    def mock_base_cache(self):
        """Create a mock base cache."""
        cache = Mock()
        cache.enabled = True
        cache.index = {}
        cache.get_caption = Mock(return_value=None)
        cache.store = Mock(return_value=True)
        return cache
    
    @pytest.fixture
    def enhanced_cache(self, mock_base_cache):
        """Create an enhanced cache with mocked base."""
        return EnhancedCaptionCache(
            mock_base_cache,
            enable_compression=True,
            compression_level=6
        )
    
    def test_initialization(self, mock_base_cache):
        """Test cache initialization."""
        cache = EnhancedCaptionCache(mock_base_cache)
        assert cache.base == mock_base_cache
        assert cache.enable_compression is True
        assert cache.compression_level == 6
        assert cache.metrics is not None
    
    def test_make_key(self, enhanced_cache):
        """Test cache key generation."""
        key = enhanced_cache._make_key("dQw4w9WgXcQ", "en")
        assert key == "dQw4w9WgXcQ_en"
    
    def test_compression_reduces_size(self, enhanced_cache):
        """Test that compression actually reduces data size."""
        # Create sample caption data
        data = {
            'video_id': 'test123',
            'segments': [
                {'text': 'This is a test caption ' * 100, 'start': i * 3.0, 'end': i * 3.0 + 2.5}
                for i in range(100)
            ]
        }
        
        compressed = enhanced_cache._compress_data(data)
        original = json.dumps(data).encode('utf-8')
        
        # Check compression marker
        assert compressed[0] in (0, 1)  # 0 = uncompressed, 1 = compressed
        
        # Should use compression for repetitive data
        if compressed[0] == 1:
            assert len(compressed) < len(original)
    
    def test_decompress_data(self, enhanced_cache):
        """Test data decompression."""
        data = {'video_id': 'test', 'text': 'Hello world'}
        
        # Compress and decompress
        compressed = enhanced_cache._compress_data(data)
        decompressed = enhanced_cache._decompress_data(compressed)
        
        assert decompressed == data
    
    def test_get_caption_miss(self, enhanced_cache, mock_base_cache):
        """Test cache miss handling."""
        mock_base_cache.get_caption.return_value = None
        
        result = enhanced_cache.get_caption("dQw4w9WgXcQ", "en")
        
        assert result is None
        assert enhanced_cache.metrics.misses == 1
        assert enhanced_cache.metrics.hits == 0
    
    def test_get_caption_hit(self, enhanced_cache, mock_base_cache):
        """Test cache hit handling."""
        mock_caption = Mock()
        mock_caption.segments = []
        mock_caption.language = "en"
        mock_base_cache.get_caption.return_value = mock_caption
        
        result = enhanced_cache.get_caption("dQw4w9WgXcQ", "en")
        
        assert result == mock_caption
        assert enhanced_cache.metrics.hits == 1
        assert enhanced_cache.metrics.misses == 0
    
    def test_get_captions_bulk(self, enhanced_cache, mock_base_cache):
        """Test bulk cache retrieval."""
        mock_caption = Mock()
        mock_caption.segments = []
        mock_caption.language = "en"
        
        # First two hits, third miss
        mock_base_cache.get_caption.side_effect = [
            mock_caption, mock_caption, None
        ]
        
        results = enhanced_cache.get_captions_bulk(
            ["vid1", "vid2", "vid3"],
            language="en"
        )
        
        assert len(results) == 3
        assert results["vid1"] == mock_caption
        assert results["vid2"] == mock_caption
        assert results["vid3"] is None
        assert enhanced_cache.metrics.bulk_hits == 2
    
    def test_get_cache_coverage(self, enhanced_cache, mock_base_cache):
        """Test cache coverage calculation."""
        mock_base_cache.index = {
            "vid1_en": {},
            "vid2_en": {},
        }
        
        coverage = enhanced_cache.get_cache_coverage(
            ["vid1", "vid2", "vid3", "vid4"],
            language="en"
        )
        
        assert coverage['total'] == 4
        assert coverage['cached'] == 2
        assert coverage['missing'] == 2
        assert coverage['coverage_ratio'] == 0.5
        assert "vid3" in coverage['missing_ids']
        assert "vid4" in coverage['missing_ids']
    
    def test_get_cache_coverage_empty(self, enhanced_cache):
        """Test cache coverage with empty list."""
        coverage = enhanced_cache.get_cache_coverage([], language="en")
        
        assert coverage['total'] == 0
        assert coverage['cached'] == 0
        assert coverage['coverage_ratio'] == 0.0
    
    def test_get_stats(self, enhanced_cache, mock_base_cache):
        """Test statistics retrieval."""
        mock_base_cache.get_stats.return_value = {'entries': 10}
        enhanced_cache.metrics.record_hit()
        
        stats = enhanced_cache.get_stats()
        
        assert 'base_cache' in stats
        assert 'enhanced' in stats
        assert stats['enhanced']['hits'] == 1
    
    def test_clear_warmed(self, enhanced_cache):
        """Test clearing warmed video tracking."""
        enhanced_cache._warmed_videos.add("test123")
        assert len(enhanced_cache._warmed_videos) == 1
        
        enhanced_cache.clear_warmed()
        assert len(enhanced_cache._warmed_videos) == 0


class TestEnhancedCacheIntegration:
    """Integration tests for enhanced cache."""
    
    def test_cache_warming_background(self):
        """Test background cache warming."""
        mock_base = Mock()
        mock_base.enabled = True
        mock_base.index = {"vid1_en": {}}
        mock_base.get_caption.return_value = Mock(segments=[], language="en")
        
        cache = EnhancedCaptionCache(mock_base)
        
        # Start background warming
        result = cache.warm_cache_for_videos(
            ["vid1", "vid2"],
            language="en",
            background=True
        )
        
        assert result['status'] == 'background_warming_started'
        assert result['videos'] == 2
    
    def test_cache_warming_sync(self):
        """Test synchronous cache warming."""
        mock_base = Mock()
        mock_base.enabled = True
        mock_base.index = {"vid1_en": {}, "vid2_en": {}}
        # Return mock for vid1 and vid2, None for vid3
        mock_base.get_caption.side_effect = [
            Mock(segments=[], language="en"),  # vid1
            Mock(segments=[], language="en"),  # vid2
            None,  # vid3
        ]
        
        cache = EnhancedCaptionCache(mock_base)
        
        result = cache.warm_cache_for_videos(
            ["vid1", "vid2", "vid3"],
            language="en",
            background=False
        )
        
        assert result['warmed'] == 2
        assert result['missed'] == 1
        assert result['hit_rate'] == 2/3
        assert 'elapsed_ms' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
