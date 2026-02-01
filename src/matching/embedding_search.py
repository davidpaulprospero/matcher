"""
Embedding-based similarity search for video segment candidates.

Extracted from main.py for single responsibility: embedding similarity search.
Provides EmbeddingSearch class for finding top-k similar video segments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, Any, Optional
import logging

from ..embeddings import find_top_k_similar

if TYPE_CHECKING:
    from ..utils import SRTSegment
    from ..config import Config

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingSearchConfig:
    """Configuration for embedding search."""
    embedding_candidates: int = 20  # Number of candidates to retrieve


class EmbeddingSearch:
    """
    Embedding-based similarity search for video segments.

    Wraps find_top_k_similar to provide a clean API for searching
    video segment candidates based on embedding similarity.

    Attributes:
        config: Search configuration
        video_embeddings: All video segment embeddings
        video_segments: All video segment objects
        embedding_index: Optional FAISS index for fast search
    """

    def __init__(
        self,
        config: EmbeddingSearchConfig,
        video_embeddings: List[List[float]],
        video_segments: List['SRTSegment'],
        embedding_index: Optional[Any] = None
    ):
        """
        Initialize embedding search.

        Args:
            config: Search configuration with embedding_candidates count
            video_embeddings: List of embedding vectors for all video segments
            video_segments: List of video SRTSegment objects (parallel to embeddings)
            embedding_index: Optional FAISS index for fast similarity search
        """
        self.config = config
        self.video_embeddings = video_embeddings
        self.video_segments = video_segments
        self.embedding_index = embedding_index

    @classmethod
    def from_matching_config(
        cls,
        matching_config: Any,
        video_embeddings: List[List[float]],
        video_segments: List['SRTSegment'],
        embedding_index: Optional[Any] = None
    ) -> 'EmbeddingSearch':
        """
        Create EmbeddingSearch from matching config section.

        Args:
            matching_config: Config.matching section with embedding_candidates
            video_embeddings: List of embedding vectors
            video_segments: List of video segments
            embedding_index: Optional FAISS index

        Returns:
            Configured EmbeddingSearch instance
        """
        embedding_candidates = getattr(matching_config, 'embedding_candidates', 20)
        config = EmbeddingSearchConfig(embedding_candidates=embedding_candidates)
        return cls(config, video_embeddings, video_segments, embedding_index)

    def search(
        self,
        query_embedding: List[float],
        num_candidates: Optional[int] = None
    ) -> List[Tuple['SRTSegment', float]]:
        """
        Search for top-k similar video segments.

        Args:
            query_embedding: Embedding vector for query (voiceover segment)
            num_candidates: Number of candidates to retrieve (overrides config)

        Returns:
            List of (video_segment, similarity_score) tuples, sorted by similarity
        """
        k = num_candidates or self.config.embedding_candidates
        k = max(k, 20)  # Ensure minimum candidates for variety

        distances, indices = self._compute_similarity(query_embedding, k)

        # Build candidate list from indices
        candidates = [
            (self.video_segments[idx], float(distances[j]))
            for j, idx in enumerate(indices)
            if 0 <= idx < len(self.video_segments)
        ]

        return candidates

    def _compute_similarity(
        self,
        query_embedding: List[float],
        k: int
    ) -> Tuple[Any, Any]:
        """
        Compute cosine similarity between query and all video embeddings.

        Uses FAISS index if available, otherwise falls back to brute-force.

        Args:
            query_embedding: Query embedding vector
            k: Number of top results to return

        Returns:
            Tuple of (distances, indices) arrays
        """
        return find_top_k_similar(
            query_embedding,
            self.video_embeddings,
            k,
            index=self.embedding_index
        )
