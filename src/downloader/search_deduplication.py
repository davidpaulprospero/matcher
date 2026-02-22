"""
Search Result Deduplication and Freshness Scoring Module

Provides functions to detect near-duplicate videos by title similarity
and video ID-based deduplication.

This module operates on search results BEFORE download to improve
search result quality and avoid redundant processing.

Deduplication strategy:
1. Primary: Video ID deduplication (exact match)
2. Secondary: Title similarity deduplication (for videos without IDs)
3. Quality preservation: Keep result with best metadata quality
4. Multi-keyword prioritization: Videos matching multiple keywords get higher relevance
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationMetrics:
    """Metrics for tracking duplicate detection across keyword searches."""
    original_count: int = 0
    deduplicated_count: int = 0
    duplicates_removed: int = 0
    videos_multi_keyword: int = 0  # Videos that matched 2+ keywords
    multi_keyword_ids: Set[str] = field(default_factory=set)
    duplicate_rate: float = 0.0

    def calculate_rate(self) -> float:
        """Calculate duplicate detection rate as percentage."""
        if self.original_count == 0:
            return 0.0
        self.duplicate_rate = (self.duplicates_removed / self.original_count) * 100
        return self.duplicate_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'original_count': self.original_count,
            'deduplicated_count': self.deduplicated_count,
            'duplicates_removed': self.duplicates_removed,
            'videos_multi_keyword': self.videos_multi_keyword,
            'multi_keyword_ids': list(self.multi_keyword_ids),
            'duplicate_rate_percent': round(self.duplicate_rate, 2),
        }


@dataclass
class VideoQualityRank:
    """Ranking for quality comparison between duplicate videos."""
    rank: int  # Lower is better
    has_thumbnail: bool = False
    has_duration: bool = False
    has_channel_name: bool = False
    view_count: int = 0


def normalize_title(title: str) -> str:
    """Normalize title for similarity comparison.

    - Lowercase
    - Remove special characters except spaces
    - Collapse multiple spaces
    - Remove common video suffixes (part 1, official video, etc.)
    """
    if not title:
        return ""

    # Lowercase
    normalized = title.lower()

    # Remove special characters except spaces and alphanumeric
    normalized = re.sub(r'[^\w\s]', ' ', normalized)

    # Collapse multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # Remove common suffixes that don't affect similarity
    suffixes_to_remove = [
        r'\bpart\s*\d+',
        r'\bofficial\s*video',
        r'\bofficial\s*music\s*video',
        r'\blyric[s]?\s*video',
        r'\baudio\b',
        r'\bhd\b',
        r'\b4k\b',
        r'\b1080p\b',
        r'\b720p\b',
        r'\bexplicit\b',
        r'\bclean\s*version\b',
        r'\bfeat\.\s*\w+',
        r'\bft\.\s*\w+',
    ]
    for suffix in suffixes_to_remove:
        normalized = re.sub(suffix, '', normalized)

    # Collapse again after suffix removal
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    return normalized


def calculate_title_similarity(title1: str, title2: str) -> float:
    """Calculate similarity between two titles using simple token-based approach.

    Uses Jaccard similarity (intersection over union of word tokens).
    Returns a value between 0.0 (completely different) and 1.0 (identical).
    """
    if not title1 or not title2:
        return 0.0

    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)

    if not norm1 or not norm2:
        return 0.0

    # Tokenize by splitting on whitespace
    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())

    if not tokens1 or not tokens2:
        return 0.0

    # Calculate Jaccard similarity
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)

    return len(intersection) / len(union) if union else 0.0


def find_duplicate_indices(
    results: List[Dict[str, Any]],
    max_similarity: float = 0.85,
) -> List[int]:
    """Find indices of duplicate/near-duplicate videos in search results.

    Uses title similarity to detect duplicates. Keeps the first occurrence
    of each group of similar videos.

    Args:
        results: List of video result dicts with 'title' key
        max_similarity: Maximum similarity threshold (0-1) to consider as duplicates.
                       Videos with similarity >= this are considered duplicates.

    Returns:
        List of indices to remove (duplicates, keeping first occurrence)
    """
    if not results or max_similarity <= 0.0 or max_similarity > 1.0:
        return []

    indices_to_remove: List[int] = []
    titles_seen: List[Tuple[str, str]] = []  # (original_title, normalized_title)

    for i, result in enumerate(results):
        title = result.get('title', '')
        if not title:
            continue

        normalized = normalize_title(title)

        # Check against all previously seen titles
        for j, (seen_title, seen_normalized) in enumerate(titles_seen):
            similarity = calculate_title_similarity(title, seen_title)
            if similarity >= max_similarity:
                # This is a duplicate, mark for removal
                indices_to_remove.append(i)
                break

        if i not in indices_to_remove:
            titles_seen.append((title, normalized))

    return indices_to_remove


def deduplicate_results(
    results: List[Dict[str, Any]],
    max_similarity: float = 0.85,
) -> List[Dict[str, Any]]:
    """Remove duplicate/near-duplicate videos from search results.

    This function applies both:
    1. Video ID-based deduplication (primary, exact match)
    2. Title similarity deduplication (fallback for results without IDs)

    Args:
        results: List of video result dicts
        max_similarity: Maximum similarity threshold for deduplication

    Returns:
        Deduplicated list of results
    """
    # First pass: Video ID-based deduplication
    results = deduplicate_by_video_id(results)

    # Second pass: Title similarity for remaining duplicates
    indices_to_remove = find_duplicate_indices(results, max_similarity)

    if not indices_to_remove:
        return results

    # Remove duplicates (reverse order to maintain correct indices)
    for idx in sorted(indices_to_remove, reverse=True):
        removed = results.pop(idx)
        logger.debug(f"Removed duplicate: {removed.get('title', 'unknown')[:50]}")

    logger.info(f"Deduplicated {len(indices_to_remove)} results ({len(results)} remaining)")

    return results


def get_video_id_hash(video_id: Optional[str]) -> Optional[str]:
    """Generate a normalized hash for a video ID.

    Args:
        video_id: The YouTube video ID

    Returns:
        Normalized hash or None if no valid video ID
    """
    if not video_id:
        return None

    # Normalize: lowercase, strip whitespace
    normalized = video_id.lower().strip()

    if not normalized:
        return None

    # Return hash for consistent comparison
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


def calculate_quality_rank(result: Dict[str, Any]) -> VideoQualityRank:
    """Calculate quality rank for a video result.

    Higher quality = lower rank number (better).

    Args:
        result: Video result dict

    Returns:
        VideoQualityRank instance
    """
    rank = 0

    # Check for thumbnails (higher quality thumbnails = better)
    thumbnail = result.get('thumbnail') or result.get('thumbnails')
    has_thumbnail = bool(thumbnail)

    # Bonus for high-quality thumbnail variants (hqdefault > mqdefault > default > none)
    thumbnail_quality_bonus = 0
    if thumbnail:
        thumbnail_str = str(thumbnail).lower()
        if 'hqdefault' in thumbnail_str:
            thumbnail_quality_bonus = -3  # Best quality (lower rank)
        elif 'mqdefault' in thumbnail_str:
            thumbnail_quality_bonus = -2  # Medium quality
        elif 'sddefault' in thumbnail_str:
            thumbnail_quality_bonus = -1  # Standard quality

    # Check for duration (must have a non-None value)
    duration = result.get('duration') or result.get('length_seconds')
    has_duration = duration is not None

    # Check for channel name
    has_channel_name = bool(result.get('channel_name') or result.get('channel_title'))

    # View count (higher = better)
    view_count = result.get('view_count', 0) or 0

    # Calculate rank (lower is better)
    rank = 0
    if not has_thumbnail:
        rank += 10
    rank += thumbnail_quality_bonus  # Apply thumbnail quality bonus
    if not has_duration:
        rank += 10
    if not has_channel_name:
        rank += 5

    return VideoQualityRank(
        rank=rank,
        has_thumbnail=has_thumbnail,
        has_duration=has_duration,
        has_channel_name=has_channel_name,
        view_count=view_count
    )


def deduplicate_by_video_id(
    results: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Remove duplicate videos based on video ID.

    Uses video ID as primary key for exact deduplication.
    When duplicates are found, keeps the result with higher quality metadata.

    Args:
        results: List of video result dicts with 'video_id' key

    Returns:
        Deduplicated list of results
    """
    if not results:
        return []

    # Track unique video IDs and their best result
    # Key: (video_hash, first_seen_index) for order preservation
    video_id_map: Dict[str, Dict[str, Any]] = {}
    first_seen_index: Dict[str, int] = {}  # Track first occurrence index
    duplicates_removed = 0

    for idx, result in enumerate(results):
        video_id = result.get('video_id')

        if not video_id:
            # No video ID, skip this deduplication pass
            continue

        video_hash = get_video_id_hash(video_id)

        if video_hash is None:
            continue

        if video_hash not in video_id_map:
            # First occurrence - store with source query info
            result['_source_query'] = result.get('_source_query', 'unknown')
            result['_dedup_hash'] = video_hash
            video_id_map[video_hash] = result
            first_seen_index[video_hash] = idx
        else:
            # Duplicate found - compare quality and keep best
            existing = video_id_map[video_hash]
            existing_rank = calculate_quality_rank(existing)
            new_rank = calculate_quality_rank(result)

            # Lower rank is better
            if new_rank.rank < existing_rank.rank:
                # New result is better quality - log replacement
                logger.debug(
                    f"Replacing duplicate video {video_id}: "
                    f"new quality rank {new_rank.rank} < existing {existing_rank.rank}"
                )
                # Log what we're replacing
                logger.info(
                    f"Removed duplicate video_id={video_id} "
                    f"(source: {result.get('_source_query', 'unknown')}, "
                    f"kept: {existing.get('_source_query', 'unknown')})"
                )
                result['_source_query'] = result.get('_source_query', 'unknown')
                result['_dedup_hash'] = video_hash
                video_id_map[video_hash] = result
                duplicates_removed += 1
            elif new_rank.rank == existing_rank.rank:
                # Same quality - prefer the one with more views
                if new_rank.view_count > existing_rank.view_count:
                    logger.debug(
                        f"Replacing duplicate video {video_id}: "
                        f"higher view count {new_rank.view_count} > {existing_rank.view_count}"
                    )
                    result['_source_query'] = result.get('_source_query', 'unknown')
                    result['_dedup_hash'] = video_hash
                    video_id_map[video_hash] = result
                    duplicates_removed += 1
                else:
                    duplicates_removed += 1
            else:
                duplicates_removed += 1

            logger.info(
                f"Removed duplicate video_id={video_id} "
                f"(source query: {result.get('_source_query', 'unknown')}, "
                f"kept source: {existing.get('_source_query', 'unknown')})"
            )

    # Rebuild results list, preserving order by first occurrence and cleaning up internal fields
    # Sort by first_seen_index to maintain original order
    sorted_hashes = sorted(video_id_map.keys(), key=lambda h: first_seen_index.get(h, float('inf')))

    deduplicated = []
    for video_hash in sorted_hashes:
        result = video_id_map[video_hash]
        # Clean up internal fields before returning
        cleaned = {k: v for k, v in result.items() if not k.startswith('_')}
        deduplicated.append(cleaned)

    # Also add results that had no video_id (pass through)
    for result in results:
        if not result.get('video_id'):
            # Clean up internal fields
            cleaned = {k: v for k, v in result.items() if not k.startswith('_')}
            deduplicated.append(cleaned)

    if duplicates_removed > 0:
        logger.info(
            f"Video ID deduplication: removed {duplicates_removed} duplicates "
            f"({len(deduplicated)} unique videos remaining)"
        )

    return deduplicated


def parse_published_at(published_at: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 published_at string to datetime.

    Handles various formats:
    - 2024-01-15T12:00:00Z
    - 2024-01-15T12:00:00+00:00
    - 2024-01-15
    """
    if not published_at:
        return None

    try:
        # Try ISO 8601 format with timezone
        if 'Z' in published_at:
            return datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        elif '+' in published_at or published_at.count('-') > 2:
            return datetime.fromisoformat(published_at)
        else:
            # Date only
            return datetime.strptime(published_at, '%Y-%m-%d')
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to parse published_at '{published_at}': {e}")
        return None


def calculate_freshness_score(
    published_at: Optional[str],
    min_freshness_days: int = 365,
    now: Optional[datetime] = None,
) -> float:
    """Calculate freshness score based on publish date.

    Args:
        published_at: ISO 8601 publish date string
        min_freshness_days: Videos older than this get reduced score
        now: Reference time (defaults to now)

    Returns:
        Freshness score between 0.0 and 1.0
        - 1.0: Published today
        - 0.5: Published min_freshness_days ago
        - 0.0: Published very long ago
    """
    if not published_at:
        return 0.5  # Unknown date gets neutral score

    if now is None:
        now = datetime.now()

    published_date = parse_published_at(published_at)
    if not published_date:
        return 0.5  # Unparseable date gets neutral score

    # Remove timezone info for age calculation if present
    if published_date.tzinfo is not None:
        published_date = published_date.replace(tzinfo=None)

    age_days = (now - published_date).days

    if age_days < 0:
        # Future date (shouldn't happen but handle gracefully)
        return 1.0

    if min_freshness_days <= 0:
        # Freshness scoring disabled
        return 1.0

    # Calculate score: 1.0 at day 0, 0.5 at min_freshness_days, approaches 0 beyond
    if age_days == 0:
        return 1.0
    elif age_days >= min_freshness_days:
        # Linear decay from 0.5 to 0.0 over another min_freshness_days period
        additional_days = age_days - min_freshness_days
        score = max(0.0, 0.5 - (additional_days / min_freshness_days) * 0.5)
        return score
    else:
        # Linear decay from 1.0 to 0.5 over min_freshness_days
        score = 1.0 - (age_days / min_freshness_days) * 0.5
        return score


def add_freshness_scores(
    results: List[Dict[str, Any]],
    min_freshness_days: int = 365,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Add freshness_score to each result based on published_at date.

    Args:
        results: List of video result dicts
        min_freshness_days: Threshold for freshness scoring
        now: Reference time for freshness calculation

    Returns:
        Results with added 'freshness_score' field
    """
    if min_freshness_days <= 0:
        # Freshness scoring disabled
        return results

    for result in results:
        published_at = result.get('published_at')
        result['freshness_score'] = calculate_freshness_score(
            published_at, min_freshness_days, now
        )

    return results


def process_search_results(
    results: List[Dict[str, Any]],
    enable_deduplication: bool = True,
    max_title_similarity: float = 0.85,
    enable_freshness_scoring: bool = True,
    min_freshness_days: int = 365,
    source_query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Process search results with deduplication and freshness scoring.

    This is the main entry point for processing search results from
    YouTube API or yt-dlp fallback.

    Args:
        results: Raw search results from API
        enable_deduplication: Whether to remove near-duplicate titles
        max_title_similarity: Similarity threshold for deduplication
        enable_freshness_scoring: Whether to add freshness scores
        min_freshness_days: Days threshold for freshness scoring
        source_query: The search query that produced these results (for logging)

    Returns:
        Processed results with deduplication applied and freshness scores added
    """
    if not results:
        return [] if results is None else results

    original_count = len(results)

    # Tag results with source query for deduplication tracking
    if source_query:
        for result in results:
            if '_source_query' not in result:
                result['_source_query'] = source_query

    # Apply deduplication
    if enable_deduplication:
        results = deduplicate_results(results, max_title_similarity)

    # Add freshness scores
    if enable_freshness_scoring:
        results = add_freshness_scores(results, min_freshness_days)

    if original_count != len(results):
        logger.info(
            f"Search results processed: {original_count} -> {len(results)} "
            f"(dedup: {enable_deduplication}, freshness: {enable_freshness_scoring}, "
            f"source: {source_query or 'unknown'})"
        )

    return results


def deduplicate_across_keywords(
    results_by_keyword: Dict[str, List[Dict[str, Any]]],
    max_similarity: float = 0.85,
    preserve_order: bool = True,
) -> Tuple[List[Dict[str, Any]], DeduplicationMetrics]:
    """Deduplicate search results across multiple keyword searches.

    This function handles deduplication when the same video appears in results
    from multiple keyword searches. Videos that match multiple keywords are
    prioritized (higher relevance).

    Args:
        results_by_keyword: Dict mapping keyword to list of video results
        max_similarity: Maximum title similarity threshold for deduplication
        preserve_order: If True, maintain original order with multi-keyword videos first

    Returns:
        Tuple of (deduplicated results, metrics)
    """
    if not results_by_keyword:
        return [], DeduplicationMetrics()

    metrics = DeduplicationMetrics()

    # Collect all results with their source keywords
    all_results: List[Dict[str, Any]] = []
    keyword_matches: Dict[str, Set[str]] = {}  # video_id -> set of keywords

    for keyword, results in results_by_keyword.items():
        for result in results:
            video_id = result.get('video_id')
            if not video_id:
                continue

            # Track which keywords matched this video
            if video_id not in keyword_matches:
                keyword_matches[video_id] = set()
            keyword_matches[video_id].add(keyword)

            # Add keyword match count to result for prioritization
            result['_matched_keywords'] = list(keyword_matches[video_id])
            result['_keyword_count'] = len(keyword_matches[video_id])

            # Track first occurrence for ordering
            if video_id not in [r.get('video_id') for r in all_results]:
                result['_first_keyword'] = keyword
                all_results.append(result)
            else:
                # Update existing result with additional keyword info
                for existing in all_results:
                    if existing.get('video_id') == video_id:
                        existing['_matched_keywords'] = list(keyword_matches[video_id])
                        existing['_keyword_count'] = len(keyword_matches[video_id])
                        break

    metrics.original_count = len(all_results)

    # Apply video ID deduplication with keyword tracking
    deduped = deduplicate_by_video_id_with_keywords(all_results)

    # Apply title similarity deduplication
    indices_to_remove = find_duplicate_indices(deduped, max_similarity)
    for idx in sorted(indices_to_remove, reverse=True):
        removed = deduped.pop(idx)
        logger.debug(f"Removed title duplicate: {removed.get('title', 'unknown')[:50]}")

    metrics.duplicates_removed = metrics.original_count - len(deduped)
    metrics.deduplicated_count = len(deduped)

    # Count videos that matched multiple keywords
    multi_keyword_count = 0
    for result in deduped:
        keyword_count = result.get('_keyword_count', 1)
        if keyword_count > 1:
            multi_keyword_count += 1
            metrics.multi_keyword_ids.add(result.get('video_id', ''))

    metrics.videos_multi_keyword = multi_keyword_count
    metrics.calculate_rate()

    # Prioritize multi-keyword videos by moving them to front (maintaining relative order)
    if preserve_order:
        multi_keyword = [r for r in deduped if r.get('_keyword_count', 1) > 1]
        single_keyword = [r for r in deduped if r.get('_keyword_count', 1) <= 1]

        # Sort each group by first occurrence order
        multi_keyword.sort(key=lambda r: r.get('_first_keyword', ''))
        single_keyword.sort(key=lambda r: r.get('_first_keyword', ''))

        deduped = multi_keyword + single_keyword

    # Clean up internal fields
    cleaned_results = []
    for result in deduped:
        cleaned = {k: v for k, v in result.items() if not k.startswith('_')}
        cleaned_results.append(cleaned)

    logger.info(
        f"Cross-keyword deduplication: {metrics.original_count} -> {metrics.deduplicated_count} "
        f"({metrics.duplicates_removed} removed, {metrics.videos_multi_keyword} multi-keyword, "
        f"rate: {metrics.duplicate_rate:.1f}%)"
    )

    return cleaned_results, metrics


def deduplicate_by_video_id_with_keywords(
    results: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Remove duplicate videos based on video ID, tracking keyword matches.

    Like deduplicate_by_video_id but preserves keyword match information
    for prioritization.

    Args:
        results: List of video result dicts with 'video_id' key

    Returns:
        Deduplicated list of results with keyword tracking
    """
    if not results:
        return []

    # Track unique video IDs and their best result with keyword info
    video_id_map: Dict[str, Dict[str, Any]] = {}
    first_seen_index: Dict[str, int] = {}

    for idx, result in enumerate(results):
        video_id = result.get('video_id')

        if not video_id:
            continue

        if video_id not in video_id_map:
            # First occurrence
            video_id_map[video_id] = result
            first_seen_index[video_id] = idx
        else:
            # Duplicate found - merge keyword match info
            existing = video_id_map[video_id]
            existing_keywords = set(existing.get('_matched_keywords', []))
            new_keywords = set(result.get('_matched_keywords', []))

            # Merge keyword sets
            merged_keywords = existing_keywords.union(new_keywords)
            existing['_matched_keywords'] = list(merged_keywords)
            existing['_keyword_count'] = len(merged_keywords)

            # Keep higher quality result
            existing_rank = calculate_quality_rank(existing)
            new_rank = calculate_quality_rank(result)

            if new_rank.rank < existing_rank.rank:
                video_id_map[video_id] = result
            elif new_rank.rank == existing_rank.rank and new_rank.view_count > existing_rank.view_count:
                video_id_map[video_id] = result

    # Rebuild results list, preserving order
    sorted_hashes = sorted(video_id_map.keys(), key=lambda h: first_seen_index.get(h, float('inf')))

    deduplicated = []
    for video_id in sorted_hashes:
        result = video_id_map[video_id]
        result['_first_keyword'] = result.get('_first_keyword', first_seen_index.get(video_id, ''))
        deduplicated.append(result)

    return deduplicated


def aggregate_results_with_keywords(
    results_by_keyword: Dict[str, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Aggregate results from multiple keyword searches, tracking keyword matches.

    This is a simpler aggregation than deduplicate_across_keywords that just
    combines results while tracking which keywords matched each video.

    Args:
        results_by_keyword: Dict mapping keyword to list of video results

    Returns:
        Combined list with '_matched_keywords' and '_keyword_count' fields added
    """
    if not results_by_keyword:
        return []

    all_results: List[Dict[str, Any]] = []
    video_keyword_map: Dict[str, Set[str]] = {}

    # First pass: collect all results and track keyword matches
    for keyword, results in results_by_keyword.items():
        for result in results:
            video_id = result.get('video_id')
            if not video_id:
                # Include results without video_id
                all_results.append(result)
                continue

            if video_id not in video_keyword_map:
                video_keyword_map[video_id] = set()
                all_results.append(result)

            video_keyword_map[video_id].add(keyword)

    # Second pass: add keyword match info to results
    for result in all_results:
        video_id = result.get('video_id')
        if video_id and video_id in video_keyword_map:
            result['_matched_keywords'] = list(video_keyword_map[video_id])
            result['_keyword_count'] = len(video_keyword_map[video_id])

    return all_results
