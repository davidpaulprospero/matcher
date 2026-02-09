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
    max_candidates_per_source: int = 3  # Max results from same video_id (0 = disabled)
    min_candidates: int = 15  # Minimum adaptive candidate pool size
    max_candidates: int = 100  # Maximum adaptive candidate pool size
    complexity_scaling_factor: float = 0.5  # How much complexity affects pool size
    pre_fetch_multiplier: int = 2  # Fetch k*multiplier from FAISS, then diversity-filter to k (US-84-005)


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
        max_per_source_raw = getattr(matching_config, 'max_candidates_per_source', 3)
        max_per_source = max_per_source_raw if isinstance(max_per_source_raw, int) else 3
        min_candidates = getattr(matching_config, 'min_candidates', 15)
        max_candidates = getattr(matching_config, 'max_candidates', 100)
        complexity_scaling_factor = getattr(matching_config, 'complexity_scaling_factor', 0.5)
        pre_fetch_raw = getattr(matching_config, 'pre_fetch_multiplier', 2)
        pre_fetch_multiplier = max(1, int(pre_fetch_raw)) if isinstance(pre_fetch_raw, (int, float)) else 2
        config = EmbeddingSearchConfig(
            embedding_candidates=embedding_candidates,
            max_candidates_per_source=max_per_source,
            min_candidates=min_candidates,
            max_candidates=max_candidates,
            complexity_scaling_factor=complexity_scaling_factor,
            pre_fetch_multiplier=pre_fetch_multiplier,
        )
        return cls(config, video_embeddings, video_segments, embedding_index)

    def search(
        self,
        query_embedding: List[float],
        num_candidates: Optional[int] = None,
        relevance_matrix: Optional[List[List[float]]] = None,
        voiceover_chapter_index: int = -1,
        relevance_boost_weight: float = 0.1,
        voiceover_text: Optional[str] = None,
    ) -> List[Tuple['SRTSegment', float]]:
        """
        Search for top-k similar video segments.

        Args:
            query_embedding: Embedding vector for query (voiceover segment)
            num_candidates: Number of candidates to retrieve (overrides config)
            relevance_matrix: 2D list [vo_chapter][vid_chapter] of relevance scores (0-1).
                When provided with voiceover_chapter_index, applies a soft boost to
                candidates from relevant video chapters (US-71-011).
            voiceover_chapter_index: Current voiceover chapter index (-1 = no chapter)
            relevance_boost_weight: Weight for the relevance boost (default 0.1)
            voiceover_text: Original voiceover text for complexity-based pool sizing (US-84-003)

        Returns:
            List of (video_segment, similarity_score) tuples, sorted by similarity
        """
        if num_candidates is not None:
            k = num_candidates
        elif voiceover_text is not None:
            # Adaptive pool sizing based on segment complexity (US-84-003)
            complexity = self._compute_complexity_score(voiceover_text)
            base_k = self.config.embedding_candidates
            # Inverse: low complexity (simple/short) -> larger pool for broader search
            # High complexity (specific/entities) -> standard pool for precision
            inverse_complexity = 1.0 - complexity
            scaled_k = base_k * (1.0 + self.config.complexity_scaling_factor * inverse_complexity)
            k = int(max(self.config.min_candidates, min(self.config.max_candidates, scaled_k)))
        else:
            k = self.config.embedding_candidates
        k = max(k, 20)  # Ensure minimum candidates for variety

        max_per_source = self.config.max_candidates_per_source
        pre_fetch_multiplier = self.config.pre_fetch_multiplier

        # Pre-fetch k*multiplier candidates from FAISS for diversity filtering (US-84-005)
        # When dedup is active, we need extra candidates to backfill after source capping
        fetch_k = k * pre_fetch_multiplier if max_per_source > 0 else k

        distances, indices = self._compute_similarity(query_embedding, fetch_k)

        # Build candidate list from indices
        candidates = [
            (self.video_segments[idx], float(distances[j]))
            for j, idx in enumerate(indices)
            if 0 <= idx < len(self.video_segments)
        ]

        # Log source concentration ratio (US-84-005)
        self._log_source_concentration(candidates)

        # Apply chapter-constrained relevance boost (US-71-011)
        candidates = self._apply_chapter_boost(
            candidates, relevance_matrix, voiceover_chapter_index, relevance_boost_weight
        )

        # Apply source diversity deduplication (US-77-009)
        if max_per_source > 0:
            candidates = self._deduplicate_by_source(candidates, max_per_source, k)

        return candidates

    # Common location words for complexity scoring
    _LOCATION_WORDS = frozenset([
        'city', 'town', 'village', 'country', 'state', 'province', 'region',
        'mountain', 'river', 'lake', 'ocean', 'sea', 'island', 'beach',
        'street', 'road', 'bridge', 'park', 'building', 'tower', 'castle',
        'cathedral', 'temple', 'mosque', 'church', 'monument', 'palace',
        'museum', 'harbor', 'port', 'airport', 'station', 'square', 'plaza',
    ])

    def _compute_complexity_score(self, text: str) -> float:
        """
        Compute complexity score from voiceover segment text (US-84-003).

        Scores range from 0.0 (simple/generic) to 1.0 (complex/specific).
        Factors:
        - Word count: more words = more specific context
        - Entity presence: capitalized multi-word phrases suggest named entities
        - Location words: geographic specificity increases complexity

        Args:
            text: Voiceover segment text

        Returns:
            Complexity score between 0.0 and 1.0
        """
        if not text or not text.strip():
            return 0.0

        words = text.split()
        word_count = len(words)

        # Word count factor: 0.0 for <=3 words, scales to 1.0 at 15+ words
        if word_count <= 3:
            word_factor = 0.0
        elif word_count >= 15:
            word_factor = 1.0
        else:
            word_factor = (word_count - 3) / 12.0

        # Entity detection: look for capitalized words that aren't sentence-starts
        entity_count = 0
        for i, word in enumerate(words):
            if i > 0 and word and word[0].isupper() and len(word) > 1:
                entity_count += 1
        entity_factor = min(entity_count / 3.0, 1.0)

        # Location word detection
        lower_words = set(w.lower() for w in words)
        location_count = len(lower_words & self._LOCATION_WORDS)
        location_factor = min(location_count / 2.0, 1.0)

        # Weighted combination
        score = 0.4 * word_factor + 0.35 * entity_factor + 0.25 * location_factor
        return min(max(score, 0.0), 1.0)

    def _log_source_concentration(
        self,
        candidates: List[Tuple['SRTSegment', float]],
    ) -> None:
        """
        Log source concentration ratio for pre-fetch FAISS results (US-84-005).

        Concentration ratio = unique_sources / total_candidates.
        When ratio < 0.3, logs a warning indicating dominated results.
        """
        if not candidates:
            return

        total = len(candidates)
        unique_sources = set()
        for segment, _ in candidates:
            source = getattr(segment, 'source_file', '') or getattr(segment, 'video_id', '') or ''
            unique_sources.add(source)

        ratio = len(unique_sources) / total
        logger.debug(
            "Source concentration: %d unique sources / %d candidates = %.2f",
            len(unique_sources), total, ratio,
        )
        if ratio < 0.3:
            logger.warning(
                "Low source diversity in FAISS results: %.1f%% unique sources "
                "(%d/%d). Consider increasing pre_fetch_multiplier.",
                ratio * 100, len(unique_sources), total,
            )

    def _apply_chapter_boost(
        self,
        candidates: List[Tuple['SRTSegment', float]],
        relevance_matrix: Optional[List[List[float]]],
        voiceover_chapter_index: int,
        relevance_boost_weight: float,
    ) -> List[Tuple['SRTSegment', float]]:
        """
        Apply soft boost to candidates from video chapters relevant to the voiceover chapter.

        Boost = relevance_score * relevance_boost_weight, applied after FAISS retrieval.
        Candidates without chapter info receive no boost or penalty (neutral).

        Args:
            candidates: List of (segment, score) from FAISS search
            relevance_matrix: [vo_chapter][vid_chapter] relevance scores
            voiceover_chapter_index: Current voiceover chapter index (-1 = none)
            relevance_boost_weight: Weight multiplier for the boost

        Returns:
            Re-sorted candidates with boosted scores
        """
        if not relevance_matrix or voiceover_chapter_index < 0:
            return candidates

        if voiceover_chapter_index >= len(relevance_matrix):
            return candidates

        row = relevance_matrix[voiceover_chapter_index]

        boosted = []
        for segment, score in candidates:
            vid_ch = getattr(segment, 'chapter_index', -1)
            if vid_ch >= 0 and vid_ch < len(row):
                boost = row[vid_ch] * relevance_boost_weight
                boosted.append((segment, score + boost))
            else:
                # No chapter info — neutral treatment
                boosted.append((segment, score))

        # Re-sort by boosted score descending
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def _deduplicate_by_source(
        self,
        candidates: List[Tuple['SRTSegment', float]],
        max_per_source: int,
        target_count: int,
    ) -> List[Tuple['SRTSegment', float]]:
        """
        Cap candidates per video source to ensure source diversity.

        Iterates through candidates (sorted by score descending) and keeps at most
        max_per_source from each video_id. Remaining slots are filled by the next
        best candidates from other sources.

        Args:
            candidates: Sorted list of (segment, score) tuples
            max_per_source: Max candidates from same source video
            target_count: Desired total candidate count

        Returns:
            Deduplicated candidate list, up to target_count items
        """
        source_counts: dict = {}
        result = []

        for segment, score in candidates:
            source = getattr(segment, 'source_file', '') or getattr(segment, 'video_id', '') or ''
            count = source_counts.get(source, 0)
            if count < max_per_source:
                result.append((segment, score))
                source_counts[source] = count + 1
                if len(result) >= target_count:
                    break

        return result

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
