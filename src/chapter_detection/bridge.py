"""
Bridge between listicle detection and chapter detection systems.

Converts ListicleGroup objects into ChapterCandidate objects so that
downstream scoring adjustments (chapter_topic_match, chapter_source_consistency,
chapter_coherence_penalty, cross_chapter_relevance) work uniformly regardless
of whether structure came from YouTube chapters or listicle detection.

US-71-010, US-72-003
"""

import logging
from typing import List, Dict, Optional, Callable, Any
from enum import Enum

from .models import ChapterCandidate, ListicleGroup

logger = logging.getLogger(__name__)


class MergeStrategy(str, Enum):
    """Strategy for merging YouTube and listicle chapters when they overlap."""
    YOUTUBE_PRIORITY = "youtube_priority"    # Current behavior: YouTube takes precedence
    HIGHEST_CONFIDENCE = "highest_confidence"  # Whichever has higher confidence wins
    UNION = "union"  # Combine topics from both sources


def listicle_groups_to_chapters(groups: List[ListicleGroup]) -> List[ChapterCandidate]:
    """
    Convert ListicleGroup objects into ChapterCandidate objects.

    Each listicle group becomes a chapter with:
    - title derived from item_label and topic_keywords
    - topics populated from topic_keywords
    - detection_strategy set to 'listicle'
    - confidence based on presence of expected_count validation

    Args:
        groups: List of ListicleGroup objects from listicle detection

    Returns:
        List of ChapterCandidate objects
    """
    chapters = []
    for group in groups:
        # Build a descriptive title from the label and keywords
        title_parts = [group.item_label] if group.item_label else []
        if group.topic_keywords:
            title_parts.append(' '.join(group.topic_keywords[:3]))
        title = ' - '.join(title_parts) if title_parts else f"Item {group.group_id + 1}"

        # Use group confidence if explicitly set (different from default 0.7)
        # Otherwise calculate based on marker type
        marker_confidence = {
            'transition': 0.6,
            'ordinal': 0.7,
            'numbered': 0.8,
        }
        if group.confidence != 0.7:  # Explicitly set by detector
            confidence = group.confidence
        else:
            confidence = marker_confidence.get(group.marker_type, 0.7)

        # Boost to 0.85 when expected_count matches detected group count
        if (group.expected_count is not None
                and group.expected_count == len(groups)):
            confidence = max(confidence, 0.85)  # Apply boost but don't reduce if already higher

        chapter = ChapterCandidate(
            chapter_id=group.group_id,
            start_segment_idx=group.start_segment_idx,
            end_segment_idx=group.end_segment_idx,
            title=title,
            topics=list(group.topic_keywords),
            confidence=confidence,
            detection_strategy='listicle',
        )
        chapters.append(chapter)

    return chapters


def _ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """Check if two segment index ranges overlap."""
    return a_start <= b_end and b_start <= a_end


def merge_chapters(
    youtube_chapters: List[ChapterCandidate],
    listicle_chapters: List[ChapterCandidate],
    merge_strategy: str = "youtube_priority",
) -> List[ChapterCandidate]:
    """
    Merge YouTube-derived chapters with listicle-derived chapters.

    Supports three merge strategies:
    - 'youtube_priority': YouTube chapters take precedence for overlapping ranges (default)
    - 'highest_confidence': Whichever source has higher confidence wins
    - 'union': Combine topics from both sources for overlapping chapters

    Args:
        youtube_chapters: Chapters from YouTube chapter detection
        listicle_chapters: Chapters converted from listicle groups
        merge_strategy: One of 'youtube_priority', 'highest_confidence', 'union'

    Returns:
        Merged list of ChapterCandidate objects sorted by start_segment_idx,
        with chapter_id reassigned sequentially.
    """
    if not youtube_chapters and not listicle_chapters:
        return []
    if not listicle_chapters:
        return list(youtube_chapters)
    if not youtube_chapters:
        return list(listicle_chapters)

    # Validate and normalize strategy
    valid_strategies = {"youtube_priority", "highest_confidence", "union"}
    if merge_strategy not in valid_strategies:
        logger.warning(
            "Unknown merge_strategy '%s', defaulting to 'youtube_priority'. "
            "Valid strategies: %s",
            merge_strategy, valid_strategies
        )
        merge_strategy = "youtube_priority"

    if merge_strategy == "youtube_priority":
        return _merge_youtube_priority(youtube_chapters, listicle_chapters)
    elif merge_strategy == "highest_confidence":
        return _merge_highest_confidence(youtube_chapters, listicle_chapters)
    elif merge_strategy == "union":
        return _merge_union(youtube_chapters, listicle_chapters)

    # Fallback (shouldn't reach here)
    return _merge_youtube_priority(youtube_chapters, listicle_chapters)


def _merge_youtube_priority(
    youtube_chapters: List[ChapterCandidate],
    listicle_chapters: List[ChapterCandidate],
) -> List[ChapterCandidate]:
    """YouTube chapters take precedence for overlapping ranges."""
    # Start with all YouTube chapters (they take precedence)
    merged = list(youtube_chapters)
    logger.debug(
        "merge_chapters: Using 'youtube_priority' strategy with %d YouTube chapters",
        len(youtube_chapters)
    )

    # Add listicle chapters that don't overlap with any YouTube chapter
    for lc in listicle_chapters:
        overlaps = any(
            _ranges_overlap(
                lc.start_segment_idx, lc.end_segment_idx,
                yc.start_segment_idx, yc.end_segment_idx,
            )
            for yc in youtube_chapters
        )
        if not overlaps:
            merged.append(lc)
            logger.debug(
                "merge_chapters: Added non-overlapping listicle chapter '%s' (segments %d-%d)",
                lc.title, lc.start_segment_idx, lc.end_segment_idx
            )

    # Sort by start index and reassign chapter IDs
    merged.sort(key=lambda c: c.start_segment_idx)
    for i, chapter in enumerate(merged):
        chapter.chapter_id = i

    return merged


def _merge_highest_confidence(
    youtube_chapters: List[ChapterCandidate],
    listicle_chapters: List[ChapterCandidate],
) -> List[ChapterCandidate]:
    """Whichever chapter has higher confidence score wins for overlapping ranges."""
    merged = []

    # Create a list of all chapters with their source and index
    # Format: (chapter, source, is_youtube, index)
    yt_with_source = [(ch, "youtube", True, i) for i, ch in enumerate(youtube_chapters)]
    li_with_source = [(ch, "listicle", False, i) for i, ch in enumerate(listicle_chapters)]
    all_chapters = yt_with_source + li_with_source

    # Track processed indices by (is_youtube, index)
    processed_yt = set()
    processed_li = set()

    for chapter, source, is_yt, idx in all_chapters:
        # Check if already processed
        if is_yt and idx in processed_yt:
            continue
        if not is_yt and idx in processed_li:
            continue

        # Find all overlapping chapters
        overlapping = []
        for other_ch, other_source, other_is_yt, other_idx in all_chapters:
            # Check if already processed
            if other_is_yt and other_idx in processed_yt:
                continue
            if not other_is_yt and other_idx in processed_li:
                continue

            if _ranges_overlap(
                chapter.start_segment_idx, chapter.end_segment_idx,
                other_ch.start_segment_idx, other_ch.end_segment_idx,
            ):
                overlapping.append((other_ch, other_source, other_is_yt, other_idx))

        if not overlapping:
            merged.append(chapter)
            if is_yt:
                processed_yt.add(idx)
            else:
                processed_li.add(idx)
            continue

        # Find the one with highest confidence
        winner = max(overlapping, key=lambda x: x[0].confidence)
        winner_ch, winner_source, winner_is_yt, winner_idx = winner

        # Log the decision
        competitors = [(ch.title, src, ch.confidence) for ch, src, _, _ in overlapping]
        logger.debug(
            "merge_chapters: Using 'highest_confidence' strategy - "
            "'%s' (%s, conf=%.2f) wins over %s",
            winner_ch.title, winner_source, winner_ch.confidence, competitors
        )

        merged.append(winner_ch)
        if winner_is_yt:
            processed_yt.add(winner_idx)
        else:
            processed_li.add(winner_idx)
        for ch, _, other_is_yt, other_idx in overlapping:
            processed_yt.add(other_idx) if other_is_yt else processed_li.add(other_idx)

    # Sort by start index and reassign chapter IDs
    merged.sort(key=lambda c: c.start_segment_idx)
    for i, chapter in enumerate(merged):
        chapter.chapter_id = i

    return merged


def _merge_union(
    youtube_chapters: List[ChapterCandidate],
    listicle_chapters: List[ChapterCandidate],
) -> List[ChapterCandidate]:
    """Combine topics from both YouTube and listicle chapters for overlapping ranges."""
    merged = []

    # Track which listicle chapters have been merged (to avoid duplicates)
    merged_listicle_indices = set()

    for yc in youtube_chapters:
        # Check if any listicle chapter overlaps
        overlapping_listicle = []
        for i, lc in enumerate(listicle_chapters):
            if i in merged_listicle_indices:
                continue
            if _ranges_overlap(
                yc.start_segment_idx, yc.end_segment_idx,
                lc.start_segment_idx, lc.end_segment_idx,
            ):
                overlapping_listicle.append((i, lc))

        if overlapping_listicle:
            # Merge topics from YouTube and all overlapping listicle chapters
            combined_topics = list(yc.topics)
            for idx, lc in overlapping_listicle:
                # Add listicle topics that aren't already present
                for topic in lc.topics:
                    if topic not in combined_topics:
                        combined_topics.append(topic)
                merged_listicle_indices.add(idx)

            # Create merged chapter
            merged_chapter = ChapterCandidate(
                chapter_id=0,  # Will be reassigned later
                start_segment_idx=yc.start_segment_idx,
                end_segment_idx=yc.end_segment_idx,
                title=yc.title,
                topics=combined_topics,
                confidence=max(yc.confidence, max((lc.confidence for _, lc in overlapping_listicle), default=0.0)),
                detection_strategy="merged",
                boundary_reasoning=f"Merged YouTube with {len(overlapping_listicle)} listicle chapter(s)",
            )

            logger.debug(
                "merge_chapters: Using 'union' strategy - merged YouTube chapter '%s' with %d listicle chapters, "
                "combined topics: %s, confidence: %.2f",
                yc.title, len(overlapping_listicle), combined_topics, merged_chapter.confidence
            )
            merged.append(merged_chapter)
        else:
            # No overlap, just add YouTube chapter as-is
            merged.append(yc)

    # Add non-overlapping listicle chapters
    for i, lc in enumerate(listicle_chapters):
        if i not in merged_listicle_indices:
            merged.append(lc)
            logger.debug(
                "merge_chapters: Added non-overlapping listicle chapter '%s' (segments %d-%d)",
                lc.title, lc.start_segment_idx, lc.end_segment_idx
            )

    # Sort by start index and reassign chapter IDs
    merged.sort(key=lambda c: c.start_segment_idx)
    for i, chapter in enumerate(merged):
        chapter.chapter_id = i

    return merged


def build_unified_chapters(
    location_chapters: List[ChapterCandidate],
    listicle_groups: List[ListicleGroup],
    merge_strategy: str = "youtube_priority",
) -> List[ChapterCandidate]:
    """
    Build a unified chapter list from both detection sources.

    This is the main entry point for the bridge. It:
    1. Converts listicle groups to chapter format
    2. Merges with YouTube/location chapters using the specified strategy
    3. Returns a unified list usable by all chapter-aware scoring

    Args:
        location_chapters: Existing chapters from YouTube/location detection
        listicle_groups: ListicleGroup objects from listicle detection
        merge_strategy: One of 'youtube_priority', 'highest_confidence', 'union'

    Returns:
        Unified list of ChapterCandidate objects
    """
    listicle_chapters = listicle_groups_to_chapters(listicle_groups)

    if not location_chapters and not listicle_chapters:
        return []

    merged = merge_chapters(location_chapters, listicle_chapters, merge_strategy)

    yt_count = len(location_chapters) if location_chapters else 0
    listicle_count = len(listicle_chapters) if listicle_chapters else 0
    logger.info(
        "US-71-010 unified chapters (%s): %d YouTube + %d listicle -> %d merged",
        merge_strategy, yt_count, listicle_count, len(merged),
    )

    return merged


def build_segment_chapter_map(
    chapters: List[ChapterCandidate],
    source: str = "video",
) -> Dict[int, int]:
    """
    Build a mapping from segment index to chapter index.

    This function handles both voiceover and video chapters bidirectionally.
    For voiceover chapters, use source="voiceover".
    For video chapters, use source="video" (default).

    Args:
        chapters: List of ChapterCandidate objects (should be sorted by start_segment_idx)
        source: "video" or "voiceover" - identifies the source for downstream scoring

    Returns:
        Dict mapping segment_index -> chapter_index (chapter_id)
    """
    mapping: Dict[int, int] = {}
    for chapter in chapters:
        for seg_idx in range(chapter.start_segment_idx, chapter.end_segment_idx + 1):
            mapping[seg_idx] = chapter.chapter_id
    return mapping


def compute_relevance_matrix(
    voiceover_chapters: List[ChapterCandidate],
    video_chapters: List[ChapterCandidate],
    embedding_fn: Optional[Callable[[str, str], float]] = None,
    keyword_weight: float = 0.6,
    embedding_weight: float = 0.4,
) -> List[List[float]]:
    """
    Compute a cross-chapter relevance matrix for candidate boosting (US-72-009).

    Each cell [i][j] is a similarity score between voiceover chapter i and
    video chapter j. When embedding_fn is provided, the score is a weighted
    blend: keyword_weight * Jaccard keyword similarity + embedding_weight * embedding cosine similarity.
    When embedding_fn is None, pure Jaccard similarity is used.

    Args:
        voiceover_chapters: Voiceover ChapterCandidate objects with topics lists
        video_chapters: Video ChapterCandidate objects with topics lists
        embedding_fn: Optional callable(text_a, text_b) -> float cosine similarity
                      in [0.0, 1.0]. When provided, blends with Jaccard.
        keyword_weight: Weight for Jaccard keyword similarity (default 0.6, US-135-002)
        embedding_weight: Weight for embedding cosine similarity (default 0.4, US-135-002)

    Returns:
        2D list of floats (vo_chapters x video_chapters), each in [0.0, 1.0].
        Empty list if either input is empty.
    """
    if not voiceover_chapters or not video_chapters:
        return []

    # Extract keyword lists from ChapterCandidate.topics
    vo_keywords = [ch.topics if ch.topics else [] for ch in voiceover_chapters]
    vid_keywords = [ch.topics if ch.topics else [] for ch in video_chapters]

    # Build chapter text representations for embedding similarity
    if embedding_fn is not None:
        vo_texts = [' '.join(kw) for kw in vo_keywords]
        vid_texts = [' '.join(kw) for kw in vid_keywords]

    # Compute matrix with optional embedding blend
    matrix = []
    for i, vo_kw in enumerate(vo_keywords):
        vo_set = {k.lower() for k in vo_kw} if vo_kw else set()
        row = []
        for j, vid_kw in enumerate(vid_keywords):
            vid_set = {k.lower() for k in vid_kw} if vid_kw else set()
            union = vo_set | vid_set
            jaccard = (len(vo_set & vid_set) / len(union)) if union else 0.0

            if embedding_fn is not None:
                cosine_sim = embedding_fn(vo_texts[i], vid_texts[j])
                score = keyword_weight * jaccard + embedding_weight * cosine_sim
            else:
                score = jaccard

            row.append(score)
        matrix.append(row)

    return matrix


def compute_chapter_alignment_scores(
    voiceover_chapters: List[ChapterCandidate],
    video_chapters: List[ChapterCandidate],
    embedding_fn: Optional[Callable[[str, str], float]] = None,
    keyword_weight: float = 0.5,
    temporal_weight: float = 0.3,
    confidence_weight: float = 0.2,
    embedding_keyword_weight: float = 0.6,
    embedding_weight: float = 0.4,
) -> Dict[str, Any]:
    """
    Compute bidirectional alignment scores between voiceover and video chapters.

    This function creates a similarity matrix considering:
    - Topic keyword overlap (Jaccard similarity)
    - Temporal alignment (segment range overlap)
    - Marker confidence (voiceover chapter detection confidence)

    Args:
        voiceover_chapters: Voiceover ChapterCandidate objects (from listicle detection)
        video_chapters: Video ChapterCandidate objects (from YouTube chapter detection)
        embedding_fn: Optional callable(text_a, text_b) -> float cosine similarity
                      in [0.0, 1.0]. When provided, blends with Jaccard.
        keyword_weight: Weight for keyword similarity (default 0.5, US-135-002)
        temporal_weight: Weight for temporal alignment (default 0.3, US-135-002)
        confidence_weight: Weight for confidence score (default 0.2, US-135-002)
        embedding_keyword_weight: Weight for Jaccard when blending with embedding (default 0.6, US-135-002)
        embedding_weight: Weight for embedding similarity (default 0.4, US-135-002)

    Returns:
        Dict containing:
        - similarity_matrix: 2D list (vo_chapters x video_chapters)
        - best_video_chapter_per_vo: List of best video chapter indices for each VO chapter
        - temporal_scores: 2D list of temporal alignment scores
        - keyword_scores: 2D list of keyword overlap scores
        - confidence_weights: List of confidence scores per VO chapter
    """
    if not voiceover_chapters or not video_chapters:
        return {
            'similarity_matrix': [],
            'best_video_chapter_per_vo': [],
            'temporal_scores': [],
            'keyword_scores': [],
            'confidence_weights': [],
        }

    # Build matrices
    keyword_scores = []
    temporal_scores = []
    confidence_weights = []

    # Extract text for embedding similarity if provided
    if embedding_fn is not None:
        vo_texts = [ch.title + ' ' + ' '.join(ch.topics) for ch in voiceover_chapters]
        vid_texts = [ch.title + ' ' + ' '.join(ch.topics) for ch in video_chapters]

    for i, vo_ch in enumerate(voiceover_chapters):
        vo_keywords = set(k.lower() for k in vo_ch.topics) if vo_ch.topics else set()
        vo_start, vo_end = vo_ch.start_segment_idx, vo_ch.end_segment_idx

        # Confidence from this voiceover chapter (affects all scores)
        conf = vo_ch.confidence if vo_ch.confidence else 0.8
        confidence_weights.append(conf)

        keyword_row = []
        temporal_row = []

        for j, vid_ch in enumerate(video_chapters):
            # Keyword overlap (Jaccard)
            vid_keywords = set(k.lower() for k in vid_ch.topics) if vid_ch.topics else set()
            union = vo_keywords | vid_keywords
            jaccard = len(vo_keywords & vid_keywords) / len(union) if union else 0.0

            if embedding_fn is not None:
                cosine_sim = embedding_fn(vo_texts[i], vid_texts[j])
                keyword_score = embedding_keyword_weight * jaccard + embedding_weight * cosine_sim
            else:
                keyword_score = jaccard

            keyword_row.append(keyword_score)

            # Temporal alignment (segment range overlap)
            vid_start, vid_end = vid_ch.start_segment_idx, vid_ch.end_segment_idx

            # Calculate overlap
            overlap_start = max(vo_start, vid_start)
            overlap_end = min(vo_end, vid_end)
            overlap = max(0, overlap_end - overlap_start)

            # Calculate temporal alignment score
            vo_range = vo_end - vo_start + 1
            vid_range = vid_end - vid_start + 1
            max_range = max(vo_range, vid_range)

            temporal_score = overlap / max_range if max_range > 0 else 0.0
            temporal_row.append(temporal_score)

        keyword_scores.append(keyword_row)
        temporal_scores.append(temporal_row)

    # Compute combined similarity matrix
    similarity_matrix = []
    best_video_chapter_per_vo = []

    for i in range(len(voiceover_chapters)):
        row = []
        for j in range(len(video_chapters)):
            score = (
                keyword_weight * keyword_scores[i][j] +
                temporal_weight * temporal_scores[i][j] +
                confidence_weight * confidence_weights[i]
            )
            row.append(score)
        similarity_matrix.append(row)

        # Find best video chapter for this VO chapter
        if row:
            best_idx = max(range(len(row)), key=lambda j: row[j])
            best_video_chapter_per_vo.append(best_idx)
        else:
            best_video_chapter_per_vo.append(-1)

    return {
        'similarity_matrix': similarity_matrix,
        'best_video_chapter_per_vo': best_video_chapter_per_vo,
        'temporal_scores': temporal_scores,
        'keyword_scores': keyword_scores,
        'confidence_weights': confidence_weights,
    }


def _compute_overlap(seg_start: float, seg_end: float, ch_start: float, ch_end: float) -> float:
    """Compute the overlap duration between a segment and a chapter time range."""
    overlap_start = max(seg_start, ch_start)
    overlap_end = min(seg_end, ch_end)
    return max(0.0, overlap_end - overlap_start)


def assign_chapter_indices(
    segments: List,
    chapters: List[Dict],
    strategy: str = 'best_match',
    adaptive_short_threshold: float = 0.25,
    adaptive_long_threshold: float = 0.75,
) -> None:
    """
    Assign chapter_index and chapter_title to each TranscriptSegment by timestamp overlap.

    Strategy options for segments spanning multiple chapters:
    - 'first': Assign to the first chapter the segment overlaps with
    - 'best_match': Assign to chapter with greatest overlap duration (default)
    - 'split': Assign to chapter where segment's midpoint falls (US-105-009)
    - 'adaptive': Choose best strategy based on segment/chapter duration ratio (US-135-012)
      - Short segment (ratio < short_threshold): 'first'
      - Long segment (ratio > long_threshold): 'split'
      - Medium segment: 'best_match'

    Segments outside all chapter ranges get chapter_index=None, chapter_title=''.

    Mutates segments in-place.

    Args:
        segments: List of TranscriptSegment objects (must have start_time, end_time)
        chapters: List of chapter dicts with keys: title, start_time, end_time.
                  Chapters are assumed to be sorted by start_time.
        strategy: Assignment strategy - 'first', 'best_match', 'split', or 'adaptive'
        adaptive_short_threshold: Ratio below which segment is considered short (default 0.25)
        adaptive_long_threshold: Ratio above which segment is considered long (default 0.75)

    US-72-003, US-105-009, US-135-012
    """
    if not chapters:
        return

    # Validate strategy
    valid_strategies = {'first', 'best_match', 'split', 'adaptive'}
    if strategy not in valid_strategies:
        logger.warning(
            "Unknown assign_chapter_indices strategy '%s', using 'best_match'. "
            "Valid strategies: %s",
            strategy, valid_strategies,
        )
        strategy = 'best_match'

    # Pre-compute chapter durations for adaptive strategy
    chapter_durations = []
    for i, ch in enumerate(chapters):
        ch_start = ch.get('start_time', 0.0)
        ch_end = ch.get('end_time', 0.0)
        if ch_end > ch_start:
            chapter_durations.append(ch_end - ch_start)
        elif i + 1 < len(chapters):
            # Use next chapter's start as end
            chapter_durations.append(chapters[i + 1].get('start_time', ch_start + 60.0) - ch_start)
        else:
            # Last chapter: assume 60 seconds
            chapter_durations.append(60.0)

    for seg in segments:
        # Determine effective strategy for adaptive mode
        effective_strategy = strategy
        if strategy == 'adaptive':
            seg_duration = seg.end_time - seg.start_time
            # Find overlapping chapters and compute average chapter duration
            overlapping_chapters = []
            for i, ch in enumerate(chapters):
                ch_start = ch.get('start_time', 0.0)
                ch_end = ch.get('end_time', 0.0)
                overlap = _compute_overlap(seg.start_time, seg.end_time, ch_start, ch_end)
                if overlap > 0.0 and i < len(chapter_durations):
                    overlapping_chapters.append(chapter_durations[i])

            if overlapping_chapters:
                avg_chapter_duration = sum(overlapping_chapters) / len(overlapping_chapters)
                if avg_chapter_duration > 0:
                    ratio = seg_duration / avg_chapter_duration
                    if ratio < adaptive_short_threshold:
                        effective_strategy = 'first'
                    elif ratio > adaptive_long_threshold:
                        effective_strategy = 'split'
                    else:
                        effective_strategy = 'best_match'
                else:
                    effective_strategy = 'best_match'
            else:
                effective_strategy = 'best_match'

        if effective_strategy == 'first':
            # Assign to first chapter with any overlap
            best_idx: Optional[int] = None
            for i, ch in enumerate(chapters):
                ch_start = ch.get('start_time', 0.0)
                ch_end = ch.get('end_time', 0.0)
                overlap = _compute_overlap(seg.start_time, seg.end_time, ch_start, ch_end)
                if overlap > 0.0:
                    best_idx = i
                    break

            if best_idx is not None:
                seg.chapter_index = best_idx
                seg.chapter_title = chapters[best_idx].get('title', '')
            else:
                seg.chapter_index = None
                seg.chapter_title = ''

        elif effective_strategy == 'split':
            # Assign to chapter where segment's midpoint falls
            # This is different from 'first' (which takes earliest overlap)
            # and 'best_match' (which takes most overlap)
            seg_midpoint = (seg.start_time + seg.end_time) / 2.0

            best_idx = None
            best_overlap = 0.0

            # First pass: find chapter where midpoint falls
            for i, ch in enumerate(chapters):
                ch_start = ch.get('start_time', 0.0)
                ch_end = ch.get('end_time', float('inf'))  # Allow open-ended chapters
                # Check if midpoint falls within chapter bounds
                if ch_start <= seg_midpoint <= ch_end:
                    best_idx = i
                    # Calculate overlap
                    actual_ch_end = ch.get('end_time', 0.0)
                    if actual_ch_end == 0.0:
                        actual_ch_end = float('inf')
                    best_overlap = _compute_overlap(seg.start_time, seg.end_time, ch_start, actual_ch_end)
                    break

            # If midpoint doesn't fall in any chapter, fall back to best_match
            if best_idx is None:
                for i, ch in enumerate(chapters):
                    ch_start = ch.get('start_time', 0.0)
                    ch_end = ch.get('end_time', 0.0)
                    overlap = _compute_overlap(seg.start_time, seg.end_time, ch_start, ch_end)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_idx = i

            if best_idx is not None and best_overlap > 0.0:
                seg.chapter_index = best_idx
                seg.chapter_title = chapters[best_idx].get('title', '')
            else:
                seg.chapter_index = None
                seg.chapter_title = ''

        else:  # 'best_match'
            best_idx = None
            best_overlap = 0.0
            seg_duration = seg.end_time - seg.start_time

            for i, ch in enumerate(chapters):
                ch_start = ch.get('start_time', 0.0)
                ch_end = ch.get('end_time', 0.0)
                overlap = _compute_overlap(seg.start_time, seg.end_time, ch_start, ch_end)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i

            # Only assign if there's meaningful overlap (>0)
            if best_idx is not None and best_overlap > 0.0:
                seg.chapter_index = best_idx
                seg.chapter_title = chapters[best_idx].get('title', '')
            else:
                seg.chapter_index = None
                seg.chapter_title = ''
