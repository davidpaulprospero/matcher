#!/usr/bin/env python3
"""
Embeddings Module - v2.5 (Consolidated)

Features:
=========
1. BATCH EMBEDDING: Groups texts into batches of 100 for Gemini API calls.
2. SMART CACHING: Embeddings cached per-video (not per-session).
3. INCREMENTAL EMBEDDING: Only computes embeddings for new segments.
4. RATE LIMIT HANDLING: Automatic retry with exponential backoff.
5. NUMPY ARRAYS: Returns numpy arrays for FAISS compatibility.
6. FAISS INTEGRATION: Proper 2D array handling for FAISS searches.
"""

import os
import json
import hashlib
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass

from .cache import BaseCache, CacheEntry, compute_hash as cache_compute_hash, batch_hash as cache_batch_hash

logger = logging.getLogger(__name__)

# Try to import numpy early
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Batch sizes for different providers (FALLBACK if config not provided)
# Primary source is config.embedding.batch_size
BATCH_SIZES = {
    'gemini': 100,      # Gemini supports up to 100 texts per call
    'voyage': 128,      # Voyage supports up to 128
    'openai': 2048,     # OpenAI supports large batches
    'local': 32,        # Local models - limited by memory
}

# Provider maximum batch sizes (hard limits from APIs)
PROVIDER_MAX_BATCH_SIZES = {
    'gemini': 100,      # Gemini API hard limit
    'voyage': 128,      # Voyage AI hard limit
    'openai': 2048,     # OpenAI limit
    'local': 256,       # Reasonable memory limit
}


def validate_batch_size(batch_size: int, provider: str = 'gemini') -> Optional[str]:
    """
    Validate batch size against provider limits.

    Args:
        batch_size: The batch size to validate
        provider: The embedding provider name ('gemini', 'voyage', 'openai', 'local')

    Returns:
        Warning message string if batch_size exceeds provider maximum, None otherwise

    Example:
        >>> validate_batch_size(150, 'gemini')
        'Batch size 150 exceeds gemini maximum of 100. Consider reducing batch_size in config.'
        >>> validate_batch_size(100, 'gemini')
        None
    """
    provider_lower = provider.lower()
    max_batch = PROVIDER_MAX_BATCH_SIZES.get(provider_lower, 100)

    if batch_size > max_batch:
        warning = (
            f"Batch size {batch_size} exceeds {provider_lower} maximum of {max_batch}. "
            f"Consider reducing batch_size in config."
        )
        logger.warning(warning)
        return warning

    return None


# Global reference for cleanup
_local_embedding_model = None


def _to_numpy(embeddings: Union[List, Any]) -> Any:
    """Convert embeddings to numpy array for FAISS compatibility"""
    if not HAS_NUMPY:
        return embeddings
    if isinstance(embeddings, np.ndarray):
        return embeddings.astype('float32')
    if isinstance(embeddings, list):
        return np.array(embeddings, dtype='float32')
    return embeddings


def cleanup_embeddings():
    """
    Unload local embedding models and free memory.

    Call this after batch embedding is complete to reclaim memory.
    Note: API-based providers (Gemini, Voyage) don't hold GPU memory.
    """
    global _local_embedding_model
    import gc

    if _local_embedding_model is not None:
        logger.info("Unloading local embedding model to free memory...")
        del _local_embedding_model
        _local_embedding_model = None

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared CUDA cache after embedding cleanup")
        except ImportError:
            pass

        logger.info("Embedding model unloaded")


def cosine_similarity(a: Any, b: Any) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector (list or numpy array)
        b: Second vector (list or numpy array)
    
    Returns:
        Cosine similarity score between -1 and 1
    """
    import numpy as np
    
    # Convert to numpy
    if isinstance(a, list):
        a = np.array(a, dtype='float32')
    if isinstance(b, list):
        b = np.array(b, dtype='float32')
    
    # Flatten if needed
    a = a.flatten()
    b = b.flatten()
    
    # Compute cosine similarity
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))


class EmbeddingCache(BaseCache):
    """Persistent embedding cache with incremental updates"""

    def __init__(self, cache_dir: str):
        cache_path = Path(cache_dir) / "embeddings"
        super().__init__(
            cache_dir=cache_path,
            index_name="embedding_index.json",
            ttl_seconds=0,  # No expiration for embeddings
            auto_save=True
        )

    def _serialize_entry(self, entry: CacheEntry) -> dict:
        """Serialize embedding cache entry"""
        return {
            'data': entry.data,
            'cached_at': entry.cached_at,
            'metadata': entry.metadata
        }

    def _deserialize_entry(self, data: dict) -> CacheEntry:
        """Deserialize embedding cache entry"""
        return CacheEntry(
            data=data['data'],
            cached_at=data['cached_at'],
            key='',
            metadata=data.get('metadata', {})
        )

    def _text_hash(self, text: str) -> str:
        """Get hash for a text string"""
        return cache_compute_hash(text, length=12)

    def _batch_hash(self, texts: List[str]) -> str:
        """Get hash for a batch of texts"""
        return cache_batch_hash(texts, length=16)
    
    def get_cached_embeddings(
        self, 
        texts: List[str],
        cache_key: str = "default"
    ) -> Tuple[List[List[float]], List[str], List[int]]:
        """
        Get cached embeddings for texts.
        
        Returns:
            (cached_embeddings, uncached_texts, uncached_indices)
        """
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text_hash = self._text_hash(text)
            cache_file = self.cache_dir / f"{cache_key}_{text_hash}.json"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        cached_embeddings.append((i, data['embedding']))
                except Exception:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        return cached_embeddings, uncached_texts, uncached_indices
    
    def cache_embeddings(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        indices: List[int],
        cache_key: str = "default"
    ):
        """Cache embeddings for texts"""
        for text, embedding, idx in zip(texts, embeddings, indices):
            text_hash = self._text_hash(text)
            cache_file = self.cache_dir / f"{cache_key}_{text_hash}.json"
            
            # Convert numpy to list for JSON serialization
            if HAS_NUMPY and isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            try:
                with open(cache_file, 'w') as f:
                    json.dump({
                        'text_preview': text[:100],
                        'embedding': embedding,
                        'cached_at': time.time()
                    }, f)
            except Exception as e:
                logger.debug(f"Could not cache embedding: {e}")
        
        # Update index
        self.index[cache_key] = {
            'count': len(texts),
            'updated_at': time.time()
        }
        self._save_index()
    
    def get_batch_cache(
        self,
        texts: List[str],
        cache_key: str = "default"
    ) -> Optional[Any]:
        """Get cached embeddings for entire batch (faster than individual lookup)"""
        batch_hash = self._batch_hash(texts)
        cache_file = self.cache_dir / f"batch_{cache_key}_{batch_hash}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    if len(data['embeddings']) == len(texts):
                        logger.info(f"  ✓ Loaded batch cache: {len(texts)} embeddings")
                        # Convert to numpy for FAISS
                        return _to_numpy(data['embeddings'])
            except Exception:
                pass
        return None
    
    def cache_batch(
        self,
        texts: List[str],
        embeddings: Any,
        cache_key: str = "default"
    ):
        """Cache entire batch of embeddings"""
        batch_hash = self._batch_hash(texts)
        cache_file = self.cache_dir / f"batch_{cache_key}_{batch_hash}.json"

        # Convert numpy to list for JSON serialization
        embeddings_list = embeddings
        if HAS_NUMPY and isinstance(embeddings, np.ndarray):
            embeddings_list = embeddings.tolist()

        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'count': len(texts),
                    'embeddings': embeddings_list,
                    'cached_at': time.time()
                }, f)
            logger.debug(f"Cached batch: {len(texts)} embeddings")
        except Exception as e:
            logger.warning(f"Could not cache batch: {e}")

    def cache_incremental(
        self,
        batch_idx: int,
        texts: List[str],
        embeddings: List[List[float]],
        cache_key: str = "default"
    ):
        """
        Cache a single batch of embeddings incrementally.

        Used for streaming mode to write progress to disk as batches complete.
        Reduces memory pressure for very large embedding jobs.
        """
        batch_file = self.cache_dir / f"incremental_{cache_key}_{batch_idx:04d}.json"

        try:
            with open(batch_file, 'w') as f:
                json.dump({
                    'batch_idx': batch_idx,
                    'texts': texts,
                    'embeddings': embeddings,
                    'cached_at': time.time()
                }, f)
        except Exception as e:
            logger.warning(f"Could not cache incremental batch {batch_idx}: {e}")

    def load_incremental(self, cache_key: str = "default") -> Tuple[List[str], List[List[float]]]:
        """
        Load all incremental batch files and combine them.

        Returns:
            (all_texts, all_embeddings) combined from batch files
        """
        all_texts = []
        all_embeddings = []

        # Find all incremental files for this cache key
        pattern = f"incremental_{cache_key}_*.json"
        batch_files = sorted(self.cache_dir.glob(pattern))

        for batch_file in batch_files:
            try:
                with open(batch_file, 'r') as f:
                    data = json.load(f)
                    all_texts.extend(data.get('texts', []))
                    all_embeddings.extend(data.get('embeddings', []))
            except Exception as e:
                logger.warning(f"Could not load incremental batch {batch_file}: {e}")

        return all_texts, all_embeddings

    def clear_incremental(self, cache_key: str = "default"):
        """Remove incremental batch files after successful consolidation"""
        pattern = f"incremental_{cache_key}_*.json"
        for batch_file in self.cache_dir.glob(pattern):
            try:
                batch_file.unlink()
            except Exception:
                pass


class EmbeddingProvider:
    """Base class for embedding providers"""
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError
    
    def embed_batch(
        self, 
        texts: List[str], 
        batch_size: int = 100,
        show_progress: bool = True,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> List[List[float]]:
        """Embed texts in batches with progress reporting
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch (from config)
            show_progress: Whether to log progress
            max_retries: Number of retry attempts (from config)
            retry_delay: Base delay between retries in seconds (from config)
        """
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            if show_progress:
                logger.info(f"    Batch {batch_num}/{total_batches} ({len(batch)} texts)")
            
            # Retry logic with exponential backoff
            for attempt in range(max_retries):
                try:
                    embeddings = self.embed(batch)
                    all_embeddings.extend(embeddings)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = retry_delay * (2 ** attempt)
                        logger.warning(f"    Batch failed, retrying in {wait:.1f}s: {e}")
                        time.sleep(wait)
                    else:
                        logger.error(f"    Batch failed after {max_retries} attempts: {e}")
                        # Fill with zeros to maintain alignment
                        # Use first embedding dimension or default to 768
                        dim = len(all_embeddings[0]) if all_embeddings else 768
                        all_embeddings.extend([[0.0] * dim] * len(batch))
        
        return all_embeddings


class GeminiEmbeddings(EmbeddingProvider):
    """Google Gemini embeddings with batching"""
    
    def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = model
        self.genai = genai
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts (up to 100)"""
        # Clean texts - Gemini doesn't like empty strings
        cleaned = [t.strip() if t.strip() else "[empty]" for t in texts]
        
        result = self.genai.embed_content(
            model=self.model,
            content=cleaned,
            task_type="retrieval_document"
        )
        
        # Handle both single and batch results
        if isinstance(result['embedding'][0], list):
            return result['embedding']
        else:
            return [result['embedding']]


class VoyageEmbeddings(EmbeddingProvider):
    """Voyage AI embeddings"""
    
    def __init__(self, api_key: str, model: str = "voyage-2"):
        import voyageai
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        cleaned = [t.strip() if t.strip() else "[empty]" for t in texts]
        result = self.client.embed(cleaned, model=self.model)
        return result.embeddings


class LocalEmbeddings(EmbeddingProvider):
    """Local sentence-transformers embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        global _local_embedding_model
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        # Store reference for cleanup
        _local_embedding_model = self.model

    def embed(self, texts: List[str]) -> List[List[float]]:
        cleaned = [t.strip() if t.strip() else "[empty]" for t in texts]
        embeddings = self.model.encode(cleaned, show_progress_bar=False)
        return embeddings.tolist()


def get_embedding_provider(config: Any) -> EmbeddingProvider:
    """Get the best available embedding provider"""
    provider_name = getattr(config.embedding, 'provider', 'gemini')
    
    if provider_name == 'gemini':
        api_key = os.getenv('GEMINI_API_KEY') or getattr(config, 'gemini_api_key', None)
        if api_key:
            try:
                model = getattr(config.embedding, 'gemini_model', 'models/text-embedding-004')
                return GeminiEmbeddings(api_key, model)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini: {e}")
    
    if provider_name == 'voyage':
        api_key = os.getenv('VOYAGE_API_KEY') or getattr(config, 'voyage_api_key', None)
        if api_key:
            try:
                model = getattr(config.embedding, 'voyage_model', 'voyage-2')
                return VoyageEmbeddings(api_key, model)
            except Exception as e:
                logger.warning(f"Could not initialize Voyage: {e}")
    
    # Fallback to local
    try:
        model = getattr(config.embedding, 'local_model', 'all-MiniLM-L6-v2')
        logger.info(f"Using local embeddings: {model}")
        return LocalEmbeddings(model)
    except Exception as e:
        logger.error(f"Could not initialize any embedding provider: {e}")
        raise RuntimeError("No embedding provider available")


def compute_embeddings(
    texts: List[str],
    provider: EmbeddingProvider,
    cache: Any,
    cache_key: str = "segments",
    show_progress: bool = True,
    config: Any = None
) -> Any:
    """
    Compute embeddings with batch processing and caching.
    
    Args:
        texts: List of text strings to embed
        provider: EmbeddingProvider instance
        cache: CacheManager or similar with cache_dir attribute
        cache_key: Key for caching (e.g., "video_segments", "voiceover")
        show_progress: Whether to show progress logs
        config: Configuration object for batch_size, retries, etc.
    
    Returns:
        Numpy array of embedding vectors (or list if numpy unavailable)
    """
    if not texts:
        return _to_numpy([])
    
    # Clean texts
    cleaned_texts = []
    for t in texts:
        t = t.strip() if t else ""
        cleaned_texts.append(t if t else "[silence]")
    
    # Get cache directory
    cache_dir = cache.cache_dir if hasattr(cache, 'cache_dir') else str(cache)
    embedding_cache = EmbeddingCache(cache_dir)

    # Check individual text cache (survives text changes between runs)
    cached_results, uncached_texts, uncached_indices = embedding_cache.get_cached_embeddings(
        cleaned_texts, cache_key
    )

    # If all texts are cached, return immediately
    if not uncached_texts:
        embeddings = [None] * len(cleaned_texts)
        for idx, emb in cached_results:
            embeddings[idx] = emb
        if show_progress:
            logger.info(f"  Loaded {len(cleaned_texts)} embeddings from cache")
        return _to_numpy(embeddings)

    # Get batch size from config (with provider-specific fallbacks)
    if config and hasattr(config, 'embedding'):
        batch_size = getattr(config.embedding, 'batch_size', 100)
        max_retries = getattr(config.embedding, 'max_retries', 3)
        retry_delay = getattr(config.embedding, 'retry_delay', 2.0)
    else:
        # Fallback based on provider type
        provider_name = type(provider).__name__.lower()
        if 'gemini' in provider_name:
            batch_size = BATCH_SIZES['gemini']
        elif 'voyage' in provider_name:
            batch_size = BATCH_SIZES['voyage']
        elif 'local' in provider_name:
            batch_size = BATCH_SIZES['local']
        else:
            batch_size = 100
        max_retries = 3
        retry_delay = 2.0

    # Compute embeddings only for uncached texts, caching incrementally per batch
    if show_progress:
        cache_pct = len(cached_results) * 100 // len(cleaned_texts) if cleaned_texts else 0
        logger.info(f"  Computing embeddings: {len(uncached_texts)}/{len(cleaned_texts)} texts ({cache_pct}% cached)")

    start_time = time.time()
    new_embeddings = []
    total_batches = (len(uncached_texts) + batch_size - 1) // batch_size

    for i in range(0, len(uncached_texts), batch_size):
        batch_texts = uncached_texts[i:i + batch_size]
        batch_indices = uncached_indices[i:i + batch_size]
        batch_num = i // batch_size + 1

        if show_progress:
            logger.info(f"    Batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")

        # Compute this batch with retries
        for attempt in range(max_retries):
            try:
                batch_embeddings = provider.embed(batch_texts)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = retry_delay * (2 ** attempt)
                    logger.warning(f"    Batch failed, retrying in {wait:.1f}s: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"    Batch failed after {max_retries} attempts: {e}")
                    dim = len(new_embeddings[0]) if new_embeddings else 768
                    batch_embeddings = [[0.0] * dim] * len(batch_texts)

        # Cache this batch immediately (survives interruption)
        embedding_cache.cache_embeddings(batch_texts, batch_embeddings, batch_indices, cache_key)
        new_embeddings.extend(batch_embeddings)

    elapsed = time.time() - start_time

    if show_progress:
        rate = len(uncached_texts) / elapsed if elapsed > 0 else 0
        logger.info(f"  Computed {len(new_embeddings)} new embeddings in {elapsed:.1f}s ({rate:.0f}/sec)")

    # Merge cached and new embeddings in correct order
    embeddings = [None] * len(cleaned_texts)
    for idx, emb in cached_results:
        embeddings[idx] = emb
    for emb, idx in zip(new_embeddings, uncached_indices):
        embeddings[idx] = emb

    # Convert to numpy for FAISS compatibility
    return _to_numpy(embeddings)


def build_embedding_index(
    embeddings: Any,
    config: Any
) -> Any:
    """Build FAISS index for fast similarity search

    Uses config.indexing settings:
        - use_faiss: Whether to use FAISS
        - index_type: 'flat' or 'ivf'
        - ivf_nlist: Number of clusters for IVF
        - ivf_nprobe: Clusters to search at query time

    Logs build time in milliseconds with vector count, dimension, and index type.
    """
    if not getattr(config.indexing, 'use_faiss', True):
        return None

    try:
        import numpy as np
        import faiss

        # Convert to numpy array if needed
        if isinstance(embeddings, list):
            embeddings_np = np.array(embeddings, dtype='float32')
        else:
            embeddings_np = embeddings.astype('float32')

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_np)

        # Build index with timing
        dimension = embeddings_np.shape[1]
        vector_count = embeddings_np.shape[0]
        index_type = getattr(config.indexing, 'index_type', 'flat')

        build_start = time.perf_counter()

        if index_type == 'flat':
            index = faiss.IndexFlatIP(dimension)  # Inner product = cosine for normalized
        elif index_type == 'ivf':
            # Get IVF settings from config
            config_nlist = getattr(config.indexing, 'ivf_nlist', 100)
            config_nprobe = getattr(config.indexing, 'ivf_nprobe', 10)

            # nlist should not exceed data size / 10
            nlist = min(config_nlist, max(1, len(embeddings_np) // 10))

            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            index.train(embeddings_np)
            index.nprobe = config_nprobe  # Set search-time clusters
        else:
            index = faiss.IndexFlatIP(dimension)

        index.add(embeddings_np)

        build_elapsed_ms = (time.perf_counter() - build_start) * 1000

        logger.info(
            f"  ✓ Built FAISS index ({index_type}): {vector_count} vectors, "
            f"dim={dimension} in {build_elapsed_ms:.1f}ms"
        )
        return index

    except ImportError:
        logger.warning("FAISS not available, using brute-force search")
        return None
    except Exception as e:
        logger.warning(f"Could not build FAISS index: {e}")
        return None


def find_top_k_similar(
    query_embedding: Any,
    embeddings: Any,
    k: int,
    index: Any = None
) -> Tuple[Any, Any]:
    """
    Find top-k most similar embeddings.
    
    Args:
        query_embedding: Query vector (list or numpy array)
        embeddings: All embeddings to search (list or numpy array)
        k: Number of results to return
        index: Optional FAISS index for fast search
    
    Returns:
        (distances, indices) - both as numpy arrays
    """
    import numpy as np
    
    # Convert query to numpy array if needed
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding, dtype='float32')
    else:
        query_embedding = np.asarray(query_embedding, dtype='float32')
    
    # Ensure 2D shape for FAISS (1, dim)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    # Normalize query for cosine similarity
    norm = np.linalg.norm(query_embedding)
    if norm > 0:
        query_embedding = query_embedding / norm
    
    # Use FAISS index if available
    if index is not None:
        try:
            import faiss
            # Make a copy for FAISS (it modifies in-place)
            query_copy = query_embedding.copy()
            faiss.normalize_L2(query_copy)
            distances, indices = index.search(query_copy, k)
            return distances[0], indices[0]
        except Exception as e:
            logger.warning(f"FAISS search failed, falling back to brute force: {e}")
    
    # Fallback to brute-force search
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings, dtype='float32')
    else:
        embeddings = np.asarray(embeddings, dtype='float32')
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    embeddings_norm = embeddings / norms
    
    # Compute cosine similarity
    similarities = np.dot(embeddings_norm, query_embedding.T).flatten()
    
    # Get top-k indices
    k = min(k, len(similarities))
    if k >= len(similarities):
        top_k_indices = np.argsort(similarities)[::-1]
    else:
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
    
    top_k_distances = similarities[top_k_indices]
    
    return top_k_distances, top_k_indices
