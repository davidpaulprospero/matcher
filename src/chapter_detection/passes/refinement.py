"""
Boundary Refinement Pass

Pass 2: Adjust chapter boundaries using embedding-based semantic similarity.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from ..models import ChapterCandidate

logger = logging.getLogger(__name__)


def run_boundary_refinement(
    chapters: List[ChapterCandidate],
    segments: List[Dict[str, Any]],
    config: Any,
    embeddings: Optional[Any] = None,
) -> List[ChapterCandidate]:
    """
    Refine chapter boundaries using semantic signals.

    Args:
        chapters: Initial chapter candidates
        segments: List of segment dicts
        config: Configuration
        embeddings: Optional pre-computed embeddings (numpy array)

    Returns:
        Refined chapter candidates
    """
    if not chapters or len(chapters) < 2:
        # Nothing to refine with single chapter
        return chapters

    # Compute embeddings if not provided
    if embeddings is None:
        embeddings = compute_voiceover_embeddings(segments, config)

    if embeddings is None or len(embeddings) == 0:
        logger.warning("No embeddings available for boundary refinement")
        return chapters

    # Compute boundary scores (semantic gaps)
    boundary_scores = compute_boundary_scores(embeddings)

    # Refine each chapter boundary
    refined = []
    for i, chapter in enumerate(chapters):
        refined_chapter = _refine_chapter_boundary(
            chapter=chapter,
            prev_chapter=chapters[i - 1] if i > 0 else None,
            next_chapter=chapters[i + 1] if i < len(chapters) - 1 else None,
            boundary_scores=boundary_scores,
            segments=segments,
            search_window=3,
        )
        refined.append(refined_chapter)

    # Update confidence based on boundary clarity
    for chapter in refined:
        _update_boundary_confidence(chapter, boundary_scores)

    logger.info(f"Refined boundaries for {len(refined)} chapters")
    return refined


def compute_voiceover_embeddings(
    segments: List[Dict[str, Any]],
    config: Any
) -> Optional[Any]:
    """
    Compute embeddings for voiceover segments.

    Uses existing embedding infrastructure with caching.
    Returns numpy array of shape (num_segments, embedding_dim).
    """
    if not segments:
        return None

    try:
        from src.embeddings import compute_embeddings, get_embedding_provider

        # Get embedding provider
        provider = get_embedding_provider(config)
        if provider is None:
            logger.warning("No embedding provider available")
            return None

        # Extract texts
        texts = [seg.get('text', '') for seg in segments]

        # Get cache directory
        cache_dir = _get_cache_dir(config)

        # Compute embeddings with dedicated cache key
        embeddings = compute_embeddings(
            texts=texts,
            provider=provider,
            cache=type('Cache', (), {'cache_dir': cache_dir})(),
            cache_key="voiceover_chapters",
            show_progress=True,
            config=config,
            embed_mode="document",
        )

        logger.info(f"Computed embeddings for {len(texts)} voiceover segments")
        return embeddings

    except ImportError as e:
        logger.warning(f"Embedding module not available: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to compute voiceover embeddings: {e}")
        return None


def compute_boundary_scores(embeddings: Any) -> List[float]:
    """
    Compute semantic gap score at each segment boundary.

    Higher score = larger semantic shift = better boundary candidate.

    Args:
        embeddings: numpy array of shape (num_segments, embedding_dim)

    Returns:
        List of boundary scores (length = num_segments - 1)
    """
    try:
        import numpy as np

        if embeddings is None or len(embeddings) < 2:
            return []

        scores = []
        for i in range(len(embeddings) - 1):
            similarity = _cosine_similarity(embeddings[i], embeddings[i + 1])
            gap_score = 1.0 - similarity  # Higher = more different
            scores.append(gap_score)

        return scores

    except Exception as e:
        logger.warning(f"Failed to compute boundary scores: {e}")
        return []


def _cosine_similarity(a: Any, b: Any) -> float:
    """Compute cosine similarity between two vectors."""
    try:
        import numpy as np

        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    except Exception:
        return 0.0


def _refine_chapter_boundary(
    chapter: ChapterCandidate,
    prev_chapter: Optional[ChapterCandidate],
    next_chapter: Optional[ChapterCandidate],
    boundary_scores: List[float],
    segments: List[Dict[str, Any]],
    search_window: int = 3,
) -> ChapterCandidate:
    """
    Refine a single chapter's boundaries.

    Looks for better boundary positions within search_window of current boundary.
    """
    if not boundary_scores:
        return chapter

    new_start = chapter.start_segment_idx
    new_end = chapter.end_segment_idx

    # Refine start boundary (if not first chapter)
    if prev_chapter is not None:
        min_start = prev_chapter.end_segment_idx + 1
        search_start = max(min_start, chapter.start_segment_idx - search_window)
        search_end = min(len(boundary_scores), chapter.start_segment_idx + search_window + 1)

        if search_start < search_end:
            best_score = 0.0
            best_idx = new_start

            for idx in range(search_start, search_end):
                if idx < len(boundary_scores):
                    score = boundary_scores[idx]
                    # Also check for transition phrases
                    if idx < len(segments):
                        text = segments[idx].get('text', '').lower()
                        if _has_transition_phrase(text):
                            score += 0.2

                    if score > best_score:
                        best_score = score
                        best_idx = idx

            new_start = best_idx

    # Refine end boundary (if not last chapter)
    if next_chapter is not None:
        max_end = next_chapter.start_segment_idx - 1
        search_start = max(0, chapter.end_segment_idx - search_window)
        search_end = min(max_end + 1, chapter.end_segment_idx + search_window + 1)

        if search_start < search_end:
            best_score = 0.0
            best_idx = new_end

            for idx in range(search_start, search_end):
                if idx < len(boundary_scores):
                    score = boundary_scores[idx]
                    if score > best_score:
                        best_score = score
                        best_idx = idx

            new_end = best_idx

    # Ensure valid range
    new_start = max(0, new_start)
    new_end = max(new_start, new_end)

    # Create refined chapter
    refined = ChapterCandidate(
        chapter_id=chapter.chapter_id,
        start_segment_idx=new_start,
        end_segment_idx=new_end,
        title=chapter.title,
        topics=chapter.topics,
        location_name=chapter.location_name,
        location_type=chapter.location_type,
        visual_keywords=chapter.visual_keywords,
        context_keywords=chapter.context_keywords,
        location_data=chapter.location_data,
        confidence=chapter.confidence,
        confidence_details=chapter.confidence_details,
        detection_strategy=chapter.detection_strategy,
        boundary_reasoning=chapter.boundary_reasoning,
    )

    # Add refinement note if boundaries changed
    if new_start != chapter.start_segment_idx or new_end != chapter.end_segment_idx:
        old_range = f"{chapter.start_segment_idx}-{chapter.end_segment_idx}"
        new_range = f"{new_start}-{new_end}"
        refined.boundary_reasoning += f" [Refined: {old_range} -> {new_range}]"

    return refined


def _has_transition_phrase(text: str) -> bool:
    """Check if text contains common transition phrases."""
    phrases = [
        'now let',
        'moving on',
        'next',
        'first',
        'second',
        'third',
        'finally',
        'in conclusion',
        'to summarize',
        'turning to',
        'let\'s look at',
        'let\'s explore',
        'speaking of',
        'meanwhile',
        'on the other hand',
    ]
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in phrases)


def _update_boundary_confidence(
    chapter: ChapterCandidate,
    boundary_scores: List[float],
) -> None:
    """Update chapter confidence based on boundary clarity."""
    if not boundary_scores:
        return

    start_idx = chapter.start_segment_idx
    end_idx = chapter.end_segment_idx

    # Get boundary scores at chapter edges
    start_score = boundary_scores[start_idx] if 0 < start_idx < len(boundary_scores) else 0.5
    end_score = boundary_scores[end_idx] if 0 <= end_idx < len(boundary_scores) else 0.5

    # Average boundary clarity
    avg_boundary_score = (start_score + end_score) / 2

    # Map to confidence adjustment (-0.2 to +0.2)
    # High boundary scores (>0.3) = clearer boundaries = higher confidence
    confidence_adj = (avg_boundary_score - 0.15) * 0.5
    confidence_adj = max(-0.2, min(0.2, confidence_adj))

    # Update confidence
    new_confidence = chapter.confidence + confidence_adj
    chapter.confidence = max(0.1, min(1.0, new_confidence))

    # Store gap score in confidence details
    if chapter.confidence_details is None:
        chapter.confidence_details = {}
    chapter.confidence_details['gap_score'] = avg_boundary_score


def _get_cache_dir(config: Any) -> str:
    """Get cache directory from config."""
    if hasattr(config, 'cache'):
        if hasattr(config.cache, 'cache_dir'):
            return config.cache.cache_dir
        if isinstance(config.cache, dict):
            return config.cache.get('cache_dir', '.cache')
    if hasattr(config, 'cache_dir'):
        return config.cache_dir
    return '.cache'
