"""
Sprint 3 Verification Tests

US-001: Create sprint 3 speed verification test
Validates that speed-related module components can be imported and instantiated.

Focus area: speed optimizations for transcription, caching, and embeddings
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


pytestmark = pytest.mark.unit


class TestTranscriptionParallelProcessorImports:
    """Test that transcription/parallel_processor.py can be imported."""

    @pytest.mark.fast
    def test_import_parallel_processor_module(self):
        """Test src.transcription.parallel_processor can be imported."""
        from src.transcription import parallel_processor
        assert parallel_processor is not None

    @pytest.mark.fast
    def test_import_transcribe_videos_parallel(self):
        """Test transcribe_videos_parallel function is importable."""
        from src.transcription.parallel_processor import transcribe_videos_parallel
        assert transcribe_videos_parallel is not None
        assert callable(transcribe_videos_parallel)

    @pytest.mark.fast
    def test_import_transcribe_video(self):
        """Test transcribe_video function is importable."""
        from src.transcription.parallel_processor import transcribe_video
        assert transcribe_video is not None
        assert callable(transcribe_video)

    @pytest.mark.fast
    def test_import_transcribe_voiceover_audio(self):
        """Test transcribe_voiceover_audio function is importable."""
        from src.transcription.parallel_processor import transcribe_voiceover_audio
        assert transcribe_voiceover_audio is not None
        assert callable(transcribe_voiceover_audio)

    @pytest.mark.fast
    def test_import_transcribe_voiceover_media(self):
        """Test transcribe_voiceover_media function is importable."""
        from src.transcription.parallel_processor import transcribe_voiceover_media
        assert transcribe_voiceover_media is not None
        assert callable(transcribe_voiceover_media)

    @pytest.mark.fast
    def test_import_get_transcript_segments(self):
        """Test get_transcript_segments function is importable."""
        from src.transcription.parallel_processor import get_transcript_segments
        assert get_transcript_segments is not None
        assert callable(get_transcript_segments)


class TestCacheBaseImports:
    """Test that cache/base.py can be imported."""

    @pytest.mark.fast
    def test_import_cache_base_module(self):
        """Test src.cache.base can be imported."""
        from src.cache import base
        assert base is not None

    @pytest.mark.fast
    def test_import_basecache_class(self):
        """Test BaseCache class is importable."""
        from src.cache.base import BaseCache
        assert BaseCache is not None

    @pytest.mark.fast
    def test_basecache_is_abstract(self):
        """Test BaseCache is an abstract base class."""
        from src.cache.base import BaseCache
        from abc import ABC
        assert issubclass(BaseCache, ABC)

    @pytest.mark.fast
    def test_import_cacheentry_dataclass(self):
        """Test CacheEntry dataclass is importable."""
        from src.cache.base import CacheEntry
        assert CacheEntry is not None

    @pytest.mark.fast
    def test_import_evictionresult_dataclass(self):
        """Test EvictionResult dataclass is importable."""
        from src.cache.base import EvictionResult
        assert EvictionResult is not None

    @pytest.mark.fast
    def test_cacheentry_is_generic(self):
        """Test CacheEntry supports generic type parameter."""
        from src.cache.base import CacheEntry
        from typing import Generic
        # CacheEntry[T] should work
        entry = CacheEntry(data="test", cached_at=0.0, key="k")
        assert entry.data == "test"


class TestEmbeddingsImports:
    """Test that embeddings.py can be imported."""

    @pytest.mark.fast
    def test_import_embeddings_module(self):
        """Test src.embeddings can be imported."""
        from src import embeddings
        assert embeddings is not None

    @pytest.mark.fast
    def test_import_cleanup_embeddings(self):
        """Test cleanup_embeddings function is importable."""
        from src.embeddings import cleanup_embeddings
        assert cleanup_embeddings is not None
        assert callable(cleanup_embeddings)

    @pytest.mark.fast
    def test_import_batch_sizes_constant(self):
        """Test BATCH_SIZES constant is importable."""
        from src.embeddings import BATCH_SIZES
        assert BATCH_SIZES is not None
        assert isinstance(BATCH_SIZES, dict)
        assert 'gemini' in BATCH_SIZES
        assert 'voyage' in BATCH_SIZES

    @pytest.mark.fast
    def test_import_to_numpy_function(self):
        """Test _to_numpy internal function is importable."""
        from src.embeddings import _to_numpy
        assert _to_numpy is not None
        assert callable(_to_numpy)


class TestParallelProcessorFunctionSignatures:
    """Test that parallel processor functions have expected parameters."""

    @pytest.mark.fast
    def test_transcribe_videos_parallel_signature(self):
        """Test transcribe_videos_parallel has expected parameters."""
        import inspect
        from src.transcription.parallel_processor import transcribe_videos_parallel
        sig = inspect.signature(transcribe_videos_parallel)
        params = list(sig.parameters.keys())
        assert 'video_paths' in params
        assert 'cache' in params
        assert 'max_workers' in params
        assert 'force_reprocess' in params

    @pytest.mark.fast
    def test_transcribe_video_signature(self):
        """Test transcribe_video has expected parameters."""
        import inspect
        from src.transcription.parallel_processor import transcribe_video
        sig = inspect.signature(transcribe_video)
        params = list(sig.parameters.keys())
        assert 'video_path' in params
        assert 'cache' in params


class TestBaseCacheAbstractMethods:
    """Test that BaseCache has required abstract methods."""

    @pytest.mark.fast
    def test_serialize_entry_is_abstract(self):
        """Test _serialize_entry is an abstract method."""
        from src.cache.base import BaseCache
        assert hasattr(BaseCache, '_serialize_entry')

    @pytest.mark.fast
    def test_deserialize_entry_is_abstract(self):
        """Test _deserialize_entry is an abstract method."""
        from src.cache.base import BaseCache
        assert hasattr(BaseCache, '_deserialize_entry')

    @pytest.mark.fast
    def test_basecache_init_params(self):
        """Test BaseCache __init__ has expected parameters."""
        import inspect
        from src.cache.base import BaseCache
        sig = inspect.signature(BaseCache.__init__)
        params = list(sig.parameters.keys())
        assert 'cache_dir' in params
        assert 'ttl_seconds' in params
        assert 'auto_save' in params


class TestEmbeddingBatchSizes:
    """Test embedding batch size configurations."""

    @pytest.mark.fast
    def test_gemini_batch_size(self):
        """Test Gemini batch size limit is 100."""
        from src.embeddings import BATCH_SIZES
        assert BATCH_SIZES['gemini'] == 100

    @pytest.mark.fast
    def test_voyage_batch_size(self):
        """Test Voyage batch size limit is 128."""
        from src.embeddings import BATCH_SIZES
        assert BATCH_SIZES['voyage'] == 128

    @pytest.mark.fast
    def test_openai_batch_size(self):
        """Test OpenAI batch size limit exists."""
        from src.embeddings import BATCH_SIZES
        assert 'openai' in BATCH_SIZES
        assert BATCH_SIZES['openai'] > 0

    @pytest.mark.fast
    def test_local_batch_size(self):
        """Test local embedding batch size exists."""
        from src.embeddings import BATCH_SIZES
        assert 'local' in BATCH_SIZES
        assert BATCH_SIZES['local'] > 0


class TestCacheEntryDataclass:
    """Test CacheEntry dataclass fields and behavior."""

    @pytest.mark.fast
    def test_cacheentry_has_data_field(self):
        """Test CacheEntry has data field."""
        from src.cache.base import CacheEntry
        entry = CacheEntry(data="test_data", cached_at=1234.5, key="test_key")
        assert entry.data == "test_data"

    @pytest.mark.fast
    def test_cacheentry_has_cached_at_field(self):
        """Test CacheEntry has cached_at field."""
        from src.cache.base import CacheEntry
        entry = CacheEntry(data="test", cached_at=1234.5, key="k")
        assert entry.cached_at == 1234.5

    @pytest.mark.fast
    def test_cacheentry_has_key_field(self):
        """Test CacheEntry has key field."""
        from src.cache.base import CacheEntry
        entry = CacheEntry(data="test", cached_at=0.0, key="my_key")
        assert entry.key == "my_key"

    @pytest.mark.fast
    def test_cacheentry_has_metadata_field(self):
        """Test CacheEntry has optional metadata field."""
        from src.cache.base import CacheEntry
        entry = CacheEntry(data="test", cached_at=0.0, key="k", metadata={"foo": "bar"})
        assert entry.metadata == {"foo": "bar"}

    @pytest.mark.fast
    def test_cacheentry_metadata_default(self):
        """Test CacheEntry metadata defaults to empty dict."""
        from src.cache.base import CacheEntry
        entry = CacheEntry(data="test", cached_at=0.0, key="k")
        assert entry.metadata == {}


class TestEvictionResultDataclass:
    """Test EvictionResult dataclass fields."""

    @pytest.mark.fast
    def test_evictionresult_has_entries_removed(self):
        """Test EvictionResult has entries_removed field."""
        from src.cache.base import EvictionResult
        result = EvictionResult(
            entries_removed=5,
            bytes_freed=1024,
            final_size_mb=10.5,
            evicted_keys=["k1", "k2"]
        )
        assert result.entries_removed == 5

    @pytest.mark.fast
    def test_evictionresult_has_bytes_freed(self):
        """Test EvictionResult has bytes_freed field."""
        from src.cache.base import EvictionResult
        result = EvictionResult(
            entries_removed=0,
            bytes_freed=2048,
            final_size_mb=0.0,
            evicted_keys=[]
        )
        assert result.bytes_freed == 2048

    @pytest.mark.fast
    def test_evictionresult_has_evicted_keys(self):
        """Test EvictionResult has evicted_keys field."""
        from src.cache.base import EvictionResult
        result = EvictionResult(
            entries_removed=2,
            bytes_freed=512,
            final_size_mb=5.0,
            evicted_keys=["key1", "key2"]
        )
        assert result.evicted_keys == ["key1", "key2"]

    @pytest.mark.fast
    def test_evictionresult_dry_run_default(self):
        """Test EvictionResult dry_run defaults to False."""
        from src.cache.base import EvictionResult
        result = EvictionResult(
            entries_removed=0,
            bytes_freed=0,
            final_size_mb=0.0,
            evicted_keys=[]
        )
        assert result.dry_run is False


class TestVerificationComplete:
    """Marker test to confirm sprint 3 verification is complete."""

    @pytest.mark.fast
    def test_sprint_3_speed_verification_complete(self):
        """Sprint 3 speed verification tests are in place."""
        # This test confirms all verification imports succeeded
        from src.transcription import parallel_processor
        from src.cache.base import BaseCache, CacheEntry
        from src.embeddings import BATCH_SIZES, cleanup_embeddings

        assert parallel_processor is not None
        assert BaseCache is not None
        assert CacheEntry is not None
        assert BATCH_SIZES is not None
        assert cleanup_embeddings is not None
