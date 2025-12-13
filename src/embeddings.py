"""
Embedding providers - local and API-based
With FAISS indexing support and hybrid embedding combination
"""

import logging
import math
from typing import List, Optional, Dict, Tuple
from abc import ABC, abstractmethod

from .config import Config
from .utils import CacheManager, ProgressBar

logger = logging.getLogger(__name__)


# =============================================================================
# FAISS INDEX (Optional - falls back to numpy if not installed)
# =============================================================================

_FAISS_AVAILABLE = False
try:
    import faiss
    import numpy as np
    _FAISS_AVAILABLE = True
    logger.debug("FAISS available for fast similarity search")
except ImportError:
    logger.debug("FAISS not installed, using numpy fallback")


class EmbeddingIndex:
    """
    Fast similarity search index.
    Uses FAISS if available, otherwise falls back to numpy brute-force.
    """
    
    def __init__(self, embeddings: List[List[float]], use_faiss: bool = True, index_type: str = "flat"):
        """
        Build an index from embeddings.
        
        Args:
            embeddings: List of embedding vectors
            use_faiss: Whether to use FAISS (if available)
            index_type: "flat" (exact) or "ivf" (approximate, faster for large datasets)
        """
        self.embeddings = embeddings
        self.dimension = len(embeddings[0]) if embeddings else 0
        self.use_faiss = use_faiss and _FAISS_AVAILABLE and len(embeddings) > 0
        self.index = None
        
        if self.use_faiss:
            self._build_faiss_index(index_type)
        else:
            # Numpy fallback - precompute norms for faster cosine similarity
            self._embeddings_array = None
            if embeddings:
                import numpy as np
                self._embeddings_array = np.array(embeddings, dtype=np.float32)
                norms = np.linalg.norm(self._embeddings_array, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                self._embeddings_normalized = self._embeddings_array / norms
    
    def _build_faiss_index(self, index_type: str):
        """Build FAISS index"""
        import numpy as np
        
        embeddings_array = np.array(self.embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity (FAISS uses inner product)
        faiss.normalize_L2(embeddings_array)
        
        if index_type == "ivf" and len(self.embeddings) > 1000:
            # IVF index for large datasets
            nlist = min(100, len(self.embeddings) // 10)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings_array)
            self.index.add(embeddings_array)
            self.index.nprobe = 10  # Number of clusters to search
        else:
            # Flat index (exact search)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings_array)
        
        logger.debug(f"Built FAISS index with {len(self.embeddings)} vectors, type={index_type}")
    
    def search(self, query: List[float], k: int = 10) -> List[Tuple[int, float]]:
        """
        Find top-k most similar embeddings.
        
        Returns:
            List of (index, similarity) tuples, sorted by similarity descending
        """
        if not self.embeddings:
            return []
        
        if self.use_faiss:
            return self._search_faiss(query, k)
        else:
            return self._search_numpy(query, k)
    
    def _search_faiss(self, query: List[float], k: int) -> List[Tuple[int, float]]:
        """Search using FAISS"""
        import numpy as np
        
        query_array = np.array([query], dtype=np.float32)
        faiss.normalize_L2(query_array)
        
        k = min(k, len(self.embeddings))
        distances, indices = self.index.search(query_array, k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0:  # -1 means no result
                results.append((int(idx), float(dist)))
        
        return results
    
    def _search_numpy(self, query: List[float], k: int) -> List[Tuple[int, float]]:
        """Search using numpy (fallback)"""
        import numpy as np
        
        query_array = np.array(query, dtype=np.float32)
        query_norm = np.linalg.norm(query_array)
        if query_norm == 0:
            return [(i, 0.0) for i in range(min(k, len(self.embeddings)))]
        
        query_normalized = query_array / query_norm
        
        # Compute all cosine similarities at once
        similarities = np.dot(self._embeddings_normalized, query_normalized)
        
        # Get top-k indices
        k = min(k, len(similarities))
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
        
        return [(int(idx), float(similarities[idx])) for idx in top_k_indices]


# =============================================================================
# HYBRID EMBEDDING COMBINER
# =============================================================================

def combine_hybrid_embeddings(
    text_embedding: List[float],
    visual_embedding: Optional[List[float]],
    text_weight: float = 0.7,
    visual_weight: float = 0.3
) -> List[float]:
    """
    Combine text and visual embeddings using weighted average.
    Falls back to text-only if visual is None or mismatched dimension.
    
    Args:
        text_embedding: Text-based embedding
        visual_embedding: Visual description embedding (optional)
        text_weight: Weight for text embedding (default 0.7)
        visual_weight: Weight for visual embedding (default 0.3)
    
    Returns:
        Combined embedding (same dimension as text_embedding)
    """
    if visual_embedding is None or len(visual_embedding) == 0:
        return text_embedding
    
    if len(text_embedding) != len(visual_embedding):
        logger.debug(f"Dimension mismatch: text={len(text_embedding)}, visual={len(visual_embedding)}, using text only")
        return text_embedding
    
    # Weighted average
    combined = [
        text_weight * t + visual_weight * v
        for t, v in zip(text_embedding, visual_embedding)
    ]
    
    # Normalize the combined embedding
    norm = math.sqrt(sum(x * x for x in combined))
    if norm > 0:
        combined = [x / norm for x in combined]
    
    return combined


def compute_hybrid_embeddings(
    text_embeddings: List[List[float]],
    visual_embeddings: Optional[List[Optional[List[float]]]],
    text_weight: float = 0.7,
    visual_weight: float = 0.3
) -> List[List[float]]:
    """
    Compute hybrid embeddings for a list of texts/visuals.
    
    Args:
        text_embeddings: List of text embeddings
        visual_embeddings: List of visual embeddings (can have None entries)
        text_weight: Weight for text component
        visual_weight: Weight for visual component
    
    Returns:
        List of hybrid embeddings
    """
    if visual_embeddings is None:
        return text_embeddings
    
    hybrid = []
    for i, text_emb in enumerate(text_embeddings):
        vis_emb = visual_embeddings[i] if i < len(visual_embeddings) else None
        hybrid.append(combine_hybrid_embeddings(text_emb, vis_emb, text_weight, visual_weight))
    
    return hybrid


class EmbeddingProvider(ABC):
    """Base class for embedding providers"""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension"""
        pass


class GeminiEmbeddings(EmbeddingProvider):
    """Google Gemini embeddings"""
    
    def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = model
        self._dimension = 768
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        import google.generativeai as genai
        
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # Filter empty texts
            batch = [t if t.strip() else "[empty]" for t in batch]
            
            result = genai.embed_content(
                model=self.model,
                content=batch,
                task_type="retrieval_document"
            )
            embeddings.extend(result['embedding'])
        
        return embeddings
    
    @property
    def dimension(self) -> int:
        return self._dimension


class VoyageEmbeddings(EmbeddingProvider):
    """Voyage AI embeddings"""
    
    def __init__(self, api_key: str, model: str = "voyage-2"):
        import voyageai
        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self._dimension = 1024
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        # Filter empty texts
        texts = [t if t.strip() else "[empty]" for t in texts]
        result = self.client.embed(texts, model=self.model)
        return result.embeddings
    
    @property
    def dimension(self) -> int:
        return self._dimension


class LocalEmbeddings(EmbeddingProvider):
    """Local embeddings using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self._dimension = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Loaded local embedding model '{model_name}' on {device}")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        # Filter empty texts
        texts = [t if t.strip() else "[empty]" for t in texts]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedding_provider(config: Config) -> EmbeddingProvider:
    """Get the configured embedding provider"""
    
    provider_type = config.embedding.provider.lower()
    
    if provider_type == "local":
        return LocalEmbeddings(config.embedding.local_model)
    
    elif provider_type == "gemini":
        if not config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not set")
        return GeminiEmbeddings(config.gemini_api_key, config.embedding.gemini_model)
    
    elif provider_type == "voyage":
        if not config.voyage_api_key:
            raise ValueError("VOYAGE_API_KEY not set")
        return VoyageEmbeddings(config.voyage_api_key, config.embedding.voyage_model)
    
    else:
        # Auto-detect available provider
        if config.gemini_api_key:
            logger.info("Using Gemini embeddings (auto-detected)")
            return GeminiEmbeddings(config.gemini_api_key)
        elif config.voyage_api_key:
            logger.info("Using Voyage embeddings (auto-detected)")
            return VoyageEmbeddings(config.voyage_api_key)
        else:
            logger.info("Using local embeddings (no API keys found)")
            return LocalEmbeddings(config.embedding.local_model)


def compute_embeddings(
    texts: List[str],
    provider: EmbeddingProvider,
    cache: CacheManager,
    cache_key: str,
    show_progress: bool = True
) -> List[List[float]]:
    """
    Compute embeddings with caching.
    """
    # Filter empty texts and track indices
    valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
    valid_texts = [texts[i] for i in valid_indices]
    
    if not valid_texts:
        logger.warning("No valid texts to embed")
        return []
    
    # Create cache key
    import hashlib
    import json
    text_hash = hashlib.md5(json.dumps(valid_texts, sort_keys=True).encode()).hexdigest()[:16]
    full_cache_key = f"{cache_key}_{text_hash}"
    
    # Check cache
    cached = cache.get_embeddings(full_cache_key)
    if cached:
        logger.info(f"Loaded cached embeddings ({len(cached)} vectors)")
        return cached
    
    # Compute embeddings
    logger.info(f"Computing embeddings for {len(valid_texts)} texts...")
    
    if show_progress:
        progress = ProgressBar(len(valid_texts), "Embedding")
        
        # Process in batches for progress updates
        batch_size = 100
        embeddings = []
        
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i+batch_size]
            batch_embeddings = provider.embed(batch)
            embeddings.extend(batch_embeddings)
            progress.update(len(batch))
        
        progress.close()
    else:
        embeddings = provider.embed(valid_texts)
    
    # Cache
    cache.save_embeddings(full_cache_key, embeddings)
    logger.info(f"Cached embeddings to {full_cache_key}")
    
    return embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors"""
    import math
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def find_top_k_similar(
    query_embedding: List[float],
    corpus_embeddings: List[List[float]],
    k: int = 10,
    index: Optional[EmbeddingIndex] = None
) -> List[tuple]:
    """
    Find top-k most similar embeddings.
    Uses FAISS index if provided, otherwise brute-force.
    
    Args:
        query_embedding: Query vector
        corpus_embeddings: Corpus of vectors to search
        k: Number of results to return
        index: Optional pre-built EmbeddingIndex for faster search
    
    Returns:
        List of (index, similarity) tuples, sorted by similarity descending
    """
    if index is not None:
        return index.search(query_embedding, k)
    
    # Fallback to brute-force
    similarities = [
        (i, cosine_similarity(query_embedding, emb))
        for i, emb in enumerate(corpus_embeddings)
    ]
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


def build_embedding_index(
    embeddings: List[List[float]],
    config: Config
) -> EmbeddingIndex:
    """
    Build an embedding index for fast similarity search.
    
    Args:
        embeddings: List of embedding vectors
        config: Configuration (for FAISS settings)
    
    Returns:
        EmbeddingIndex ready for search
    """
    use_faiss = config.indexing.use_faiss
    index_type = config.indexing.index_type
    
    return EmbeddingIndex(embeddings, use_faiss=use_faiss, index_type=index_type)
