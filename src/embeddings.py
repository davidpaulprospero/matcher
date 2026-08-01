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
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass

from .cache import BaseCache, CacheEntry, compute_hash as cache_compute_hash, batch_hash as cache_batch_hash

logger = logging.getLogger(__name__)

# Import cost tracking (US-162-010)
try:
    from .llm_client.cost import calculate_embedding_cost, get_cost_tracker
    HAS_COST_TRACKING = True
except ImportError:
    HAS_COST_TRACKING = False

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
    'ollama': 50,       # Ollama - conservative for local inference
}

# Provider maximum batch sizes (hard limits from APIs)
PROVIDER_MAX_BATCH_SIZES = {
    'gemini': 100,      # Gemini API hard limit
    'voyage': 128,      # Voyage AI hard limit
    'openai': 2048,     # OpenAI limit
    'local': 256,       # Reasonable memory limit
    'ollama': 512,      # Ollama - depends on available RAM/VRAM
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
    """Convert embeddings to numpy array for FAISS compatibility.
    Drops entries with wrong dimensions (defensive against stale cache from a different model)."""
    if not HAS_NUMPY:
        return embeddings
    if isinstance(embeddings, np.ndarray):
        return embeddings.astype('float32')
    if isinstance(embeddings, list):
        # Detect expected dim from the MOST COMMON non-None length (first entry may be stale/wrong)
        from collections import Counter
        dim_counts = Counter()
        for e in embeddings:
            if e is not None and isinstance(e, list) and len(e) > 0:
                dim_counts[len(e)] += 1
        if dim_counts:
            expected_dim = dim_counts.most_common(1)[0][0]
            # Replace wrong-dim entries with zero vectors of correct dim so np.array succeeds
            dropped = 0
            for i, e in enumerate(embeddings):
                if e is not None and isinstance(e, list) and len(e) != expected_dim:
                    embeddings[i] = [0.0] * expected_dim
                    dropped += 1
            if dropped > 0:
                logger.warning(f"[EMBEDDING] _to_numpy: replaced {dropped} wrong-dim entries with zero vectors (expected dim={expected_dim}, dim_distribution={dict(dim_counts)})")
        return np.array(embeddings, dtype='float32')
    return embeddings


@dataclass
class EmbeddingValidationResult:
    """Result of embedding integrity validation."""
    total_count: int
    valid_count: int
    none_count: int
    wrong_dimension_count: int
    unnormalized_count: int
    expected_dimension: Optional[int]
    valid_indices: List[int]
    invalid_indices: List[int]

    @property
    def is_valid(self) -> bool:
        """True if all embeddings are valid."""
        return self.valid_count == self.total_count

    @property
    def all_none(self) -> bool:
        """True if every embedding is None."""
        return self.none_count == self.total_count


def validate_embedding_integrity(
    embeddings: Any,
    norm_tolerance: float = 0.1
) -> EmbeddingValidationResult:
    """
    Validate embedding integrity before matching.

    Checks:
    - No None values in the embedding list
    - Consistent dimensions across all embeddings
    - Norms approximately 1.0 (unit vectors for cosine similarity)

    Args:
        embeddings: List or numpy array of embedding vectors. Individual
            entries may be None if the embedding API failed for that segment.
        norm_tolerance: Acceptable deviation from unit norm (default 0.1,
            i.e. norms between 0.9 and 1.1 are considered normalized).

    Returns:
        EmbeddingValidationResult with counts and valid/invalid indices.
    """
    import numpy as np

    total = len(embeddings) if embeddings is not None else 0
    if total == 0:
        return EmbeddingValidationResult(
            total_count=0, valid_count=0, none_count=0,
            wrong_dimension_count=0, unnormalized_count=0,
            expected_dimension=None,
            valid_indices=[], invalid_indices=[],
        )

    # Determine expected dimension from first non-None embedding
    expected_dim = None
    for emb in embeddings:
        if emb is not None:
            try:
                vec = np.asarray(emb, dtype='float32')
                expected_dim = vec.shape[-1] if vec.ndim >= 1 else None
                break
            except Exception:
                continue

    none_count = 0
    wrong_dim_count = 0
    unnorm_count = 0
    valid_indices = []
    invalid_indices = []

    for i, emb in enumerate(embeddings):
        if emb is None:
            none_count += 1
            invalid_indices.append(i)
            continue

        try:
            vec = np.asarray(emb, dtype='float32').flatten()
        except Exception:
            invalid_indices.append(i)
            none_count += 1
            continue

        # Check dimension
        if expected_dim is not None and vec.shape[0] != expected_dim:
            wrong_dim_count += 1
            invalid_indices.append(i)
            continue

        # Check norm (unit vector for cosine similarity)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            # Zero vector — treat as invalid
            unnorm_count += 1
            invalid_indices.append(i)
            continue
        if abs(norm - 1.0) > norm_tolerance:
            unnorm_count += 1
            # Still usable but flag it — don't exclude
            valid_indices.append(i)
            continue

        valid_indices.append(i)

    valid_count = len(valid_indices)

    return EmbeddingValidationResult(
        total_count=total,
        valid_count=valid_count,
        none_count=none_count,
        wrong_dimension_count=wrong_dim_count,
        unnormalized_count=unnorm_count,
        expected_dimension=expected_dim,
        valid_indices=valid_indices,
        invalid_indices=invalid_indices,
    )


def cleanup_embeddings():
    """
    Unload local embedding models and free memory.

    Call this after batch embedding is complete to reclaim memory.
    Note: API-based providers (Gemini, Voyage) don't hold GPU memory.
    """
    global _local_embedding_model
    import gc

    if _local_embedding_model is not None:
        model_name = getattr(_local_embedding_model, 'name', 'unknown')
        logger.info(f"[EMBEDDING] Unloading model: provider=local, model={model_name}")

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

        logger.info(f"[EMBEDDING] Model unloaded: provider=local, model={model_name}")


def cosine_similarity(a: Any, b: Any, use_cache: bool = True) -> float:
    """
    Compute cosine similarity between two vectors with optional memoization.

    Uses a global cache to avoid recomputing similarity for the same
    embedding pairs across multiple strategy tracks (6+ per segment).

    Args:
        a: First vector (list or numpy array)
        b: Second vector (list or numpy array)
        use_cache: Whether to use memoization cache (default: True)

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

    # Check cache if enabled
    if use_cache:
        try:
            from .matching.similarity_cache import get_similarity_cache, embedding_hash
            cache = get_similarity_cache()
            hash_a = embedding_hash(a)
            hash_b = embedding_hash(b)

            # Check cache
            cached = cache.get(hash_a, hash_b)
            if cached is not None:
                return cached

            # Compute and cache
            result = _compute_cosine_similarity(a, b)
            cache.put(hash_a, hash_b, result)
            return result
        except ImportError:
            # Cache not available, compute directly
            pass

    return _compute_cosine_similarity(a, b)


def _compute_cosine_similarity(a: Any, b: Any) -> float:
    """
    Raw cosine similarity computation without caching.

    Args:
        a: First vector (numpy array)
        b: Second vector (numpy array)

    Returns:
        Cosine similarity score between -1 and 1
    """
    import numpy as np

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
    
    def embed(self, texts: List[str], embed_mode: str = "document") -> List[List[float]]:
        raise NotImplementedError

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = True,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        embed_mode: str = "document"
    ) -> List[List[float]]:
        """Embed texts in batches with progress reporting

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch (from config)
            show_progress: Whether to log progress
            max_retries: Number of retry attempts (from config)
            retry_delay: Base delay between retries in seconds (from config)
            embed_mode: "document" for corpus texts, "query" for search queries
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
                    embeddings = self.embed(batch, embed_mode=embed_mode)
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

    def __init__(self, api_key: str, model: str = "models/gemini-embedding-001"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = model
        self.genai = genai
        self.provider_name = "gemini"

    def embed(self, texts: List[str], embed_mode: str = "document") -> List[List[float]]:
        """Embed a batch of texts (up to 100)"""
        # Clean texts - Gemini doesn't like empty strings
        cleaned = [t.strip() if t.strip() else "[empty]" for t in texts]

        # Map embed_mode to Gemini task_type for asymmetric search
        task_type = "retrieval_query" if embed_mode == "query" else "retrieval_document"

        result = self.genai.embed_content(
            model=self.model,
            content=cleaned,
            task_type=task_type
        )

        # Track embedding cost (US-162-010)
        if HAS_COST_TRACKING:
            # Estimate token count (rough approximation: ~4 chars per token)
            estimated_tokens = sum(len(t) // 4 for t in cleaned)
            cost = calculate_embedding_cost(
                provider=self.provider_name,
                model=self.model,
                token_count=estimated_tokens
            )
            if cost > 0:
                tracker = get_cost_tracker()
                tracker.add_embedding_cost(cost=cost, provider=self.provider_name)
                logger.debug(f"Embedding API call: provider={self.provider_name}, texts={len(cleaned)}, tokens={estimated_tokens}, cost=${cost:.6f}")

        # Handle both single and batch results
        if isinstance(result['embedding'][0], list):
            embeddings = result['embedding']
        else:
            embeddings = [result['embedding']]

        # Log embedding computation details (US-169-011)
        vector_dim = len(embeddings[0]) if embeddings else 0
        logger.debug(
            f"[EMBEDDING] Computed: provider={self.provider_name}, model={self.model}, "
            f"texts={len(embeddings)}, vector_dim={vector_dim}"
        )
        return embeddings


class VoyageEmbeddings(EmbeddingProvider):
    """Voyage AI embeddings"""

    def __init__(self, api_key: str, model: str = "voyage-2"):
        import voyageai
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.provider_name = "voyage"

    def embed(self, texts: List[str], embed_mode: str = "document") -> List[List[float]]:
        cleaned = [t.strip() if t.strip() else "[empty]" for t in texts]

        # Track embedding cost (US-162-010) - before call to get estimate
        estimated_tokens = sum(len(t) // 4 for t in cleaned)

        result = self.client.embed(cleaned, model=self.model)

        # Track embedding cost (US-162-010)
        if HAS_COST_TRACKING:
            # Use actual token count from response if available
            actual_tokens = getattr(result, 'tokens_used', None) or estimated_tokens
            cost = calculate_embedding_cost(
                provider=self.provider_name,
                model=self.model,
                token_count=actual_tokens
            )
            if cost > 0:
                tracker = get_cost_tracker()
                tracker.add_embedding_cost(cost=cost, provider=self.provider_name)
                logger.debug(f"Embedding API call: provider={self.provider_name}, texts={len(cleaned)}, tokens={actual_tokens}, cost=${cost:.6f}")

        embeddings = result.embeddings

        # Log embedding computation details (US-169-011)
        vector_dim = len(embeddings[0]) if embeddings else 0
        logger.debug(
            f"[EMBEDDING] Computed: provider={self.provider_name}, model={self.model}, "
            f"texts={len(embeddings)}, vector_dim={vector_dim}"
        )
        return embeddings


class LocalEmbeddings(EmbeddingProvider):
    """Local sentence-transformers embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_retries: int = 5, retry_delay: float = 5.0):
        global _local_embedding_model
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.provider_name = "local"

        # Log model initialization
        logger.info(f"[EMBEDDING] Loading local embedding model: provider=local, model={model_name}")

        # Retry logic for model loading (handles HuggingFace network issues)
        last_error = None
        for attempt in range(max_retries):
            try:
                self.model = SentenceTransformer(model_name)
                # Store reference for cleanup
                _local_embedding_model = self.model

                # Log model loaded with memory info
                self._log_memory_usage()
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = retry_delay * (attempt + 1)
                    logger.warning(f"Model load failed (attempt {attempt + 1}/{max_retries}), retrying in {wait:.0f}s: {e}")
                    time.sleep(wait)

        raise RuntimeError(f"Failed to load SentenceTransformer after {max_retries} attempts: {last_error}")

    def _log_memory_usage(self):
        """Log memory usage for the loaded model."""
        try:
            import numpy as np

            # Estimate model size (sentence-transformers models are typically 80-500MB)
            # Get model's embedding dimension
            embedding_dim = self.model.get_sentence_embedding_dimension()

            # Rough estimate: ~4 bytes per float32 * embedding_dim * vocab_size_estimate
            # This is approximate but gives a useful relative measure
            model_params = getattr(self.model, 'max_seq_length', 256)
            estimated_size_mb = (embedding_dim * model_params * 4) / (1024 * 1024)

            # Add current process memory if available
            process = psutil.Process(sys.argv[0]) if 'psutil' in sys.modules else None
            if process:
                process_mem_mb = process.memory_info().rss / (1024 * 1024)
                logger.info(f"[EMBEDDING] Model loaded: provider=local, model={self.model_name}, embedding_dim={embedding_dim}, estimated_size={estimated_size_mb:.1f}MB, process_memory={process_mem_mb:.1f}MB")
            else:
                logger.info(f"[EMBEDDING] Model loaded: provider=local, model={self.model_name}, embedding_dim={embedding_dim}, estimated_size={estimated_size_mb:.1f}MB")
        except Exception as e:
            logger.info(f"[EMBEDDING] Model loaded: provider=local, model={self.model_name}")

    def embed(self, texts: List[str], embed_mode: str = "document") -> List[List[float]]:
        cleaned = [t.strip() if t.strip() else "[empty]" for t in texts]
        embeddings = self.model.encode(cleaned, show_progress_bar=False)

        # Log local embedding call (US-162-010) - cost is 0 for local
        # Log embedding computation details (US-169-011)
        embeddings_list = embeddings.tolist()
        vector_dim = len(embeddings_list[0]) if embeddings_list else 0
        logger.debug(
            f"[EMBEDDING] Computed: provider={self.provider_name}, model={self.model_name}, "
            f"texts={len(embeddings_list)}, vector_dim={vector_dim}"
        )

        return embeddings_list


class OllamaEmbeddings(EmbeddingProvider):
    """Ollama local embeddings (nomic-embed-text, 768-dim)

    Supports asymmetric search via task prefixes:
    - "search_document: " for corpus/document texts
    - "search_query: " for search queries
    """

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434", timeout: int = 120):
        import requests as _requests
        self._requests = _requests
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.provider_name = "ollama"
        self._verify_availability()

    def _verify_availability(self):
        """Check that Ollama server is running and model is pulled."""
        try:
            resp = self._requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except self._requests.ConnectionError:
            raise RuntimeError(
                f"Ollama server not reachable at {self.base_url}. "
                f"Start with: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama server error: {e}")

        # Check model is available
        try:
            models = resp.json().get('models', [])
            model_names = [m.get('name', '').split(':')[0] for m in models]
            if self.model not in model_names:
                raise RuntimeError(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available: {model_names}. "
                    f"Pull with: ollama pull {self.model}"
                )
        except RuntimeError:
            raise
        except Exception:
            # If we can't parse the response, just proceed — embed call will fail clearly
            logger.warning("Could not verify Ollama model availability, proceeding anyway")

    def embed(self, texts: List[str], embed_mode: str = "document") -> List[List[float]]:
        """Embed texts via Ollama /api/embed endpoint with task prefixes."""
        cleaned = [t.strip() if t.strip() else "[empty]" for t in texts]

        # Apply nomic-embed-text task prefix for asymmetric search
        prefix = "search_query: " if embed_mode == "query" else "search_document: "
        prefixed = [prefix + t for t in cleaned]

        payload = {
            "model": self.model,
            "input": prefixed,
            "keep_alive": "5m",
            "truncate": True,
        }

        try:
            resp = self._requests.post(
                f"{self.base_url}/api/embed",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data["embeddings"]

            # Log local embedding call (US-162-010) - cost is 0 for local
            # Log embedding computation details (US-169-011)
            vector_dim = len(embeddings[0]) if embeddings else 0
            logger.debug(
                f"[EMBEDDING] Computed: provider={self.provider_name}, model={self.model}, "
                f"texts={len(embeddings)}, vector_dim={vector_dim}"
            )

            return embeddings
        except self._requests.ConnectionError:
            raise RuntimeError(
                f"Ollama server not reachable at {self.base_url}. "
                f"Start with: ollama serve"
            )
        except self._requests.Timeout:
            raise RuntimeError(
                f"Ollama embedding request timed out after {self.timeout}s. "
                f"The model may be loading for the first time."
            )
        except KeyError:
            raise RuntimeError(
                f"Unexpected Ollama response format: {data}"
            )


def get_embedding_provider(config: Any) -> EmbeddingProvider:
    """Get the best available embedding provider"""
    provider_name = getattr(config.embedding, 'provider', 'gemini')

    if provider_name == 'gemini':
        api_key = os.getenv('GEMINI_API_KEY') or getattr(config, 'gemini_api_key', None)
        if api_key:
            try:
                model = getattr(config.embedding, 'gemini_model', 'models/gemini-embedding-001')
                logger.info(f"[EMBEDDING] Initializing provider: provider=gemini, model={model}")
                return GeminiEmbeddings(api_key, model)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini: {e}")

    if provider_name == 'voyage':
        api_key = os.getenv('VOYAGE_API_KEY') or getattr(config, 'voyage_api_key', None)
        if api_key:
            try:
                model = getattr(config.embedding, 'voyage_model', 'voyage-2')
                logger.info(f"[EMBEDDING] Initializing provider: provider=voyage, model={model}")
                return VoyageEmbeddings(api_key, model)
            except Exception as e:
                logger.warning(f"Could not initialize Voyage: {e}")

    if provider_name == 'ollama':
        try:
            model = getattr(config.embedding, 'ollama_model', 'nomic-embed-text')
            base_url = getattr(config.embedding, 'ollama_base_url', 'http://localhost:11434')
            logger.info(f"[EMBEDDING] Initializing provider: provider=ollama, model={model}, base_url={base_url}")
            return OllamaEmbeddings(model, base_url)
        except Exception as e:
            logger.warning(f"Could not initialize Ollama embeddings: {e}")

    # Fallback to local
    try:
        model = getattr(config.embedding, 'local_model', 'all-MiniLM-L6-v2')
        logger.info(f"[EMBEDDING] Initializing provider: provider=local, model={model}")
        return LocalEmbeddings(model)
    except Exception as e:
        logger.error(f"Could not initialize any embedding provider: {e}")
        raise RuntimeError("No embedding provider available")


def parallel_embed_batch(
    texts: List[str],
    provider: EmbeddingProvider,
    config: Any = None,
    show_progress: bool = True,
    embed_mode: str = "document"
) -> List[List[float]]:
    """
    Embed texts in parallel batches using ThreadPoolExecutor.

    Splits input texts into batches and processes them in parallel threads
    for improved performance on large text sets (200+ texts).

    Args:
        texts: List of texts to embed
        provider: EmbeddingProvider instance
        config: Configuration object with embedding.max_workers, batch_size, etc.
        show_progress: Whether to log progress
        embed_mode: "document" for corpus texts, "query" for search queries

    Returns:
        List of embedding vectors in original order

    Example:
        >>> from src.embeddings import parallel_embed_batch, get_embedding_provider
        >>> provider = get_embedding_provider(config)
        >>> embeddings = parallel_embed_batch(texts, provider, config)
        >>> len(embeddings) == len(texts)
        True
    """
    if not texts:
        return []

    # Get config values with safe defaults
    if config and hasattr(config, 'embedding'):
        batch_size = getattr(config.embedding, 'batch_size', 100)
        max_workers = getattr(config.embedding, 'max_workers', 4)
        max_retries = getattr(config.embedding, 'max_retries', 3)
        retry_delay = getattr(config.embedding, 'retry_delay', 2.0)
    else:
        batch_size = 100
        max_workers = 4
        max_retries = 3
        retry_delay = 2.0

    # Clean texts
    cleaned_texts = []
    for t in texts:
        t = t.strip() if t else ""
        cleaned_texts.append(t if t else "[silence]")

    # Split into batches
    batches = []
    for i in range(0, len(cleaned_texts), batch_size):
        batches.append((i // batch_size, cleaned_texts[i:i + batch_size]))

    total_batches = len(batches)

    logger.info(f"[EMBEDDING] Parallel batch processing: total_texts={len(cleaned_texts)}, batches={total_batches}, workers={max_workers}, batch_size={batch_size}, provider={provider.provider_name}")

    # Results storage: {batch_index: embeddings}
    results: Dict[int, List[List[float]]] = {}
    start_time = time.time()

    def process_batch(batch_tuple: Tuple[int, List[str]]) -> Tuple[int, List[List[float]]]:
        """Process a single batch with retries."""
        batch_idx, batch_texts = batch_tuple

        for attempt in range(max_retries):
            try:
                embeddings = provider.embed(batch_texts, embed_mode=embed_mode)
                return (batch_idx, embeddings)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = retry_delay * (2 ** attempt)
                    logger.warning(f"    Batch {batch_idx + 1} failed, retrying in {wait:.1f}s: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"    Batch {batch_idx + 1} failed after {max_retries} attempts: {e}")
                    # Return zeros to maintain alignment
                    dim = 768  # Default dimension
                    return (batch_idx, [[0.0] * dim] * len(batch_texts))

    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_batch, batch): batch[0] for batch in batches}

        completed = 0
        for future in as_completed(futures):
            batch_idx, embeddings = future.result()
            results[batch_idx] = embeddings
            completed += 1

            if show_progress:
                logger.info(f"    Completed batch {completed}/{total_batches}")

    # Reassemble results in original order
    all_embeddings = []
    for i in range(total_batches):
        all_embeddings.extend(results[i])

    elapsed = time.time() - start_time

    rate = len(cleaned_texts) / elapsed if elapsed > 0 else 0
    logger.info(f"[EMBEDDING] Parallel batch complete: computed={len(all_embeddings)}, elapsed={elapsed:.2f}s, rate={rate:.1f}/sec, provider={provider.provider_name}")

    return all_embeddings


def compute_embeddings(
    texts: List[str],
    provider: EmbeddingProvider,
    cache: Any,
    cache_key: str = "segments",
    show_progress: bool = True,
    config: Any = None,
    embed_mode: str = "document"
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
        embed_mode: "document" for corpus texts, "query" for search queries

    Returns:
        Numpy array of embedding vectors (or list if numpy unavailable)
    """
    # Check for test mode flag to skip embedding computation
    if config and getattr(config, '_test_mode_skip_embeddings', False):
        if show_progress:
            logger.info(f"  Test mode: returning empty embeddings for {len(texts)} texts")
        # Return empty embeddings with proper shape
        return _to_numpy([])

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

    # Provider-qualified cache key prevents cross-provider cache pollution
    provider_tag = type(provider).__name__.lower().replace("embeddings", "")
    qualified_cache_key = f"{provider_tag}_{cache_key}"

    # Check individual text cache (survives text changes between runs)
    cached_results, uncached_texts, uncached_indices = embedding_cache.get_cached_embeddings(
        cleaned_texts, qualified_cache_key
    )

    # ALSO check incremental cache (per-batch files written by previous interrupted runs)
    # These were saved as one-file-per-batch so they don't trigger the
    # per-text cache directory's inode/page-cache pressure that stalled
    # 80K+ text caches under memory pressure.
    if uncached_texts:
        try:
            incr_texts, incr_embeddings = embedding_cache.load_incremental(qualified_cache_key)
            if incr_texts:
                incr_map = {t: e for t, e in zip(incr_texts, incr_embeddings)}
                # Filter uncached_texts: anything in incremental is now cached
                still_uncached = []
                still_indices = []
                for text, idx in zip(uncached_texts, uncached_indices):
                    if text in incr_map:
                        cached_results.append((idx, incr_map[text]))
                    else:
                        still_uncached.append(text)
                        still_indices.append(idx)
                uncached_texts = still_uncached
                uncached_indices = still_indices
                if cached_results and not uncached_texts:
                    logger.info(
                        f"[EMBEDDING] Incremental cache hit: {len(cached_results)} embeddings recovered"
                    )
        except Exception as e:
            logger.debug(f"[EMBEDDING] Incremental cache load failed (non-fatal): {e}")

    # If all texts are cached, return immediately
    if not uncached_texts:
        embeddings = [None] * len(cleaned_texts)
        for idx, emb in cached_results:
            embeddings[idx] = emb
        # Sanitize: detect expected dim from MOST COMMON length (defensive against stale cache from a different model)
        from collections import Counter
        dim_counts = Counter()
        for e in embeddings:
            if e is not None and isinstance(e, list) and len(e) > 0:
                dim_counts[len(e)] += 1
        if dim_counts:
            expected_dim = dim_counts.most_common(1)[0][0]
            for i, e in enumerate(embeddings):
                if e is not None and isinstance(e, list) and len(e) != expected_dim:
                    embeddings[i] = [0.0] * expected_dim  # zero-vector placeholder
            if len(dim_counts) > 1:
                logger.warning(f"[EMBEDDING] Sanitized cache: replaced {sum(1 for e in embeddings if e is None or (isinstance(e, list) and len(e) != expected_dim))} wrong-dim entries (expected dim={expected_dim}, dim_distribution={dict(dim_counts)})")
        if show_progress:
            logger.info(f"[EMBEDDING] Cache hit: loaded {len(cleaned_texts)} embeddings from cache (100% cached)")
        logger.debug(f"[EMBEDDING] Cache hit: texts={len(cleaned_texts)}, cache_key={qualified_cache_key}")
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
    cache_pct = len(cached_results) * 100 // len(cleaned_texts) if cleaned_texts else 0
    logger.info(
        f"[EMBEDDING] Batch processing: total_texts={len(cleaned_texts)}, "
        f"cached={len(cached_results)}, uncached={len(uncached_texts)}, "
        f"cache_hit_pct={cache_pct}%, batch_size={batch_size}, provider={provider.provider_name}"
    )

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
                batch_embeddings = provider.embed(batch_texts, embed_mode=embed_mode)
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

        # Cache this batch incrementally (1 file per 100 texts).
        # This avoids writing 100 individual files per batch into a directory
        # that already has 60k+ entries — which triggers kernel page-cache
        # pressure (folio_wait_bit_common in D state) under memory-tight
        # conditions. Per-text cache files already on disk remain valid for
        # get_cached_embeddings() lookups; this gives us crash safety in
        # consolidated batches instead.
        embedding_cache.cache_incremental(
            batch_num, batch_texts, batch_embeddings, qualified_cache_key
        )
        new_embeddings.extend(batch_embeddings)

    elapsed = time.time() - start_time

    rate = len(uncached_texts) / elapsed if elapsed > 0 else 0
    logger.info(f"[EMBEDDING] Batch complete: computed={len(new_embeddings)}, elapsed={elapsed:.2f}s, rate={rate:.1f}/sec, provider={provider.provider_name}")

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
