"""
Main matching orchestration.

Migrated from matching.py (lines 2484-2872, ~390 lines).
Provides the public match_all_segments() API for voiceover-to-video matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Dict, Any, Tuple
from collections import defaultdict
from pathlib import Path
import logging

from .tracking import TimelineVarietyTracker, GlobalClipTracker
from .strategies import StrategyMatcher
from .embedding_search import EmbeddingSearch
from .candidate_filter import filter_by_context_relevance
from ..utils import SRTSegment, MatchResult, ProgressBar
from ..state import Match
from ..embeddings import validate_embedding_integrity
from ..chapter_detection.bridge import compute_relevance_matrix
from ..logging_templates import log_error_with_context

if TYPE_CHECKING:
    from ..config import Config
    from ..utils import CacheManager, SceneInfo, VideoTopics
    from ..topic_extraction import LocationChapter
    from ..location_service import GeoLocation

logger = logging.getLogger(__name__)


def _is_usable_primary_match(result: MatchResult, min_confidence: Optional[float] = None) -> bool:
    """Return True when a result has a real, non-gap V1 clip usable for output."""
    if not result:
        return False

    primary = getattr(result, 'primary_match', None)
    if not primary:
        return False

    if getattr(result, 'has_gap', False):
        return False

    if getattr(primary, 'match_type', '') == 'gap':
        return False

    video_segment = getattr(primary, 'video_segment', None)
    source_file = getattr(video_segment, 'source_file', None) if video_segment else None
    if source_file is not None and not str(source_file).strip():
        return False

    confidence = float(getattr(primary, 'confidence', 0.0))
    if min_confidence is not None and confidence < min_confidence:
        return False

    return True


def _find_segment_listicle_group(
    segment_idx: int,
    listicle_groups: List,
) -> Optional[Dict]:
    """
    Find which listicle group a voiceover segment belongs to.

    Args:
        segment_idx: Index of the voiceover segment
        listicle_groups: List of ListicleGroup objects or dicts

    Returns:
        The listicle group dict/object if found, None otherwise
    """
    if not listicle_groups:
        return None

    for group in listicle_groups:
        # Handle both dict and object formats
        start_idx = group.get('start_segment_idx') if isinstance(group, dict) else getattr(group, 'start_segment_idx', None)
        end_idx = group.get('end_segment_idx') if isinstance(group, dict) else getattr(group, 'end_segment_idx', None)

        if start_idx is not None and end_idx is not None:
            if start_idx <= segment_idx <= end_idx:
                return group

    return None


def _compute_topic_overlap(
    video_topics: List[str],
    listicle_topics: List[str],
) -> float:
    """
    Compute the overlap between video topics and listicle topics.

    Args:
        video_topics: List of topic keywords from video segment
        listicle_topics: List of topic keywords from listicle group

    Returns:
        Overlap ratio (0.0-1.0): |intersection| / |union|
    """
    if not video_topics or not listicle_topics:
        return 0.0

    # Normalize to lowercase for comparison
    video_set = set(t.lower() for t in video_topics)
    listicle_set = set(t.lower() for t in listicle_topics)

    intersection = video_set & listicle_set
    union = video_set | listicle_set

    if not union:
        return 0.0

    return len(intersection) / len(union)


def filter_candidates_by_listicle(
    candidates: List[Tuple[SRTSegment, float]],
    segment_idx: int,
    listicle_groups: List,
    boundary_strictness: str,
    min_topic_overlap: float,
) -> List[Tuple[SRTSegment, float]]:
    """
    Filter video candidates based on listicle group boundaries (US-135-006).

    Filters candidates so that video segments only match voiceover segments
    within the same listicle group (or adjacent groups in relaxed mode).

    Args:
        candidates: List of (video_segment, similarity_score) tuples
        segment_idx: Index of the voiceover segment being matched
        listicle_groups: List of ListicleGroup objects
        boundary_strictness: 'strict', 'relaxed', or 'disabled'
        min_topic_overlap: Minimum topic overlap required (0.0-1.0)

    Returns:
        Filtered list of candidates
    """
    # Disabled mode: no filtering
    if boundary_strictness == 'disabled':
        return candidates

    # No listicle groups: no filtering
    if not listicle_groups:
        return candidates

    # Find the listicle group for this segment
    current_group = _find_segment_listicle_group(segment_idx, listicle_groups)

    # No listicle group found: no filtering
    if not current_group:
        return candidates

    # Get listicle group properties
    group_id = current_group.get('group_id') if isinstance(current_group, dict) else getattr(current_group, 'group_id', 0)
    listicle_topics = current_group.get('topic_keywords', []) if isinstance(current_group, dict) else getattr(current_group, 'topic_keywords', [])

    if not listicle_topics:
        # No topics to filter by: allow all candidates
        return candidates

    # Determine allowed group IDs based on strictness
    allowed_group_ids: set = set()

    if boundary_strictness == 'strict':
        # Only allow same group
        allowed_group_ids = {group_id}
    elif boundary_strictness == 'relaxed':
        # Allow same group and adjacent groups (group_id ± 1)
        allowed_group_ids = {group_id - 1, group_id, group_id + 1}
    else:
        # Unknown strictness: allow all
        return candidates

    # Find all listicle groups and their topic keywords
    group_topics: Dict[int, List[str]] = {}
    for group in listicle_groups:
        g_id = group.get('group_id') if isinstance(group, dict) else getattr(group, 'group_id', None)
        g_topics = group.get('topic_keywords', []) if isinstance(group, dict) else getattr(group, 'topic_keywords', [])
        if g_id is not None:
            group_topics[g_id] = g_topics

    # Filter candidates
    filtered_candidates = []
    for video_seg, score in candidates:
        # Get video topics
        video_topics = video_seg.topics if hasattr(video_seg, 'topics') else []
        if not video_topics:
            # No topics: include if any group has empty topics, otherwise skip
            video_topics = video_seg.keywords if hasattr(video_seg, 'keywords') else []
            if not video_topics:
                # No topics available - include this video (topic filtering shouldn't penalize missing metadata)
                filtered_candidates.append((video_seg, score))
                continue

        # Check if video belongs to an allowed listicle group based on topic overlap
        video_allowed = False

        for allowed_id in allowed_group_ids:
            if allowed_id not in group_topics:
                continue

            overlap = _compute_topic_overlap(video_topics, group_topics[allowed_id])
            if overlap >= min_topic_overlap:
                video_allowed = True
                break

        if video_allowed:
            filtered_candidates.append((video_seg, score))

    return filtered_candidates


def match_all_segments(
    voiceover_segments: List[SRTSegment],
    video_segments: List[SRTSegment],
    voiceover_embeddings: List[List[float]],
    video_embeddings: List[List[float]],
    scenes: Optional[Dict[str, List['SceneInfo']]],
    config: 'Config',
    cache: 'CacheManager',
    embedding_index: Optional[Any] = None,
    face_preference: str = "neutral",
    video_topics: Optional[Dict[str, 'VideoTopics']] = None,
    location_chapters: Optional[List['LocationChapter']] = None,
    video_locations: Optional[Dict[str, 'GeoLocation']] = None,
    video_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    listicle_groups: Optional[List] = None,
    progress_callback: Optional[Callable[[int, List[MatchResult]], None]] = None,
    start_index: int = 0,
    prior_results: Optional[List[MatchResult]] = None,
) -> List[MatchResult]:
    """
    Match all voiceover segments to video segments.

    Migrated from matching.py lines 2484-2872.

    Processes sequentially to ensure accurate reuse tracking.
    Also computes strategy matches for V4-V10 with variety enforcement.

    Two-stage matching optimization:
    - Stage 1: Retrieve more candidates from embeddings (embedding_candidates)
    - Stage 2: Send only top candidates to LLM for reranking (llm_rerank_candidates)

    Timeline variety enforcement:
    - Prevents same source video from appearing multiple times within a time window
    - Configured via config.output.variety.timeline_variety_window (default 600s = 10 min)

    Chapter-based topic matching:
    - If video_topics provided, applies confidence penalty for topic mismatches
    - Helps ensure videos match the voiceover chapter/topic context

    Location-aware matching:
    - If location_chapters provided, filters candidates by geographic location
    - Hard filter by country with soft penalty fallback

    Args:
        voiceover_segments: List of voiceover SRTSegment objects
        video_segments: List of video SRTSegment objects
        voiceover_embeddings: Embeddings for voiceover segments
        video_embeddings: Embeddings for video segments
        scenes: Scene information dict
        config: Config object
        cache: CacheManager instance
        embedding_index: Optional FAISS index for fast similarity search
        face_preference: "more" (prefer faces), "none" (avoid faces), or "neutral"
        video_topics: Dict of video_path -> VideoTopics for chapter-based matching
        location_chapters: List of LocationChapter for location-aware matching
        video_locations: Dict of video_path -> GeoLocation for location matching
        progress_callback: Optional callback(index, results) called every N segments
        start_index: Index to resume from (skip segments before this)
        prior_results: Pre-populated results list for resumed matching

    Returns:
        List of MatchResult objects, one per voiceover segment
    """
    # Import TieredMatcher here to avoid circular import
    from .tiered_matcher import TieredMatcher, create_gap_match

    matcher = TieredMatcher(config, cache, video_topics=video_topics,
                            video_metadata=video_metadata,
                            listicle_groups=listicle_groups)

    # Set up location-aware matching if provided
    if location_chapters:
        matcher.set_location_chapters(location_chapters)
    if video_locations:
        matcher.set_video_locations(video_locations)
    strategy_matcher = StrategyMatcher(config)

    # Store face preference for use during matching
    matcher.face_preference = face_preference

    mc = config.matching
    oc = config.output
    vc = oc.variety
    include_alternatives = getattr(oc, 'include_alternatives', True)
    num_alternatives = getattr(oc, 'num_alternatives', 2) if include_alternatives else 0
    num_secondary_tracks = 3  # V4-V6 are always present
    min_candidates_for_multitrack = 1 + num_alternatives + num_secondary_tracks
    dedup_relax_threshold = max(getattr(mc, 'llm_rerank_candidates', 5), min_candidates_for_multitrack)

    # Get segment time range for logging
    if voiceover_segments:
        first_seg = voiceover_segments[0]
        last_seg = voiceover_segments[-1]
        seg_time_range = f"{first_seg.start_time:.1f}s-{last_seg.end_time:.1f}s"
    else:
        seg_time_range = "N/A"

    logger.info(f"Matching {len(voiceover_segments)} voiceover segments (time range: {seg_time_range})")
    logger.info(f"  Two-stage matching: embedding_candidates={mc.embedding_candidates}, llm_rerank={mc.llm_rerank_candidates}")
    logger.info(f"  Reuse prevention: max_reuse={mc.max_clip_reuse}, penalty={mc.reuse_penalty}")
    if mc.max_clip_reuse == 1:
        logger.info(f"  Mode: Each clip can only be used ONCE")

    if face_preference != "neutral":
        logger.info(f"  Face preference: {face_preference}")

    if mc.duration_scoring_enabled:
        logger.info(f"  Duration scoring: ideal={mc.ideal_speed_range}, soft={mc.soft_penalty_range}")

    # Timeline variety enforcement
    # Handle vc as either object or dict
    if isinstance(vc, dict):
        timeline_variety_enabled = vc.get('enforce_timeline_variety', True)
        timeline_window = vc.get('timeline_variety_window', 600.0)
        max_repeats = vc.get('max_source_repeats_in_window', 1)
    else:
        timeline_variety_enabled = getattr(vc, 'enforce_timeline_variety', True)
        timeline_window = getattr(vc, 'timeline_variety_window', 600.0)
        max_repeats = getattr(vc, 'max_source_repeats_in_window', 1)

    variety_tracker = None
    if timeline_variety_enabled:
        variety_tracker = TimelineVarietyTracker(
            timeline_window=timeline_window,
            max_repeats=max_repeats
        )
        logger.info(f"  Timeline variety: {timeline_window/60:.0f}min window, max {max_repeats} repeat(s) per source")

    # Global clip tracker for cross-segment deduplication
    global_clip_tracker = None
    clip_hard_block = getattr(mc, 'clip_hard_block', True)
    if clip_hard_block:
        global_clip_tracker = GlobalClipTracker()
        logger.info(f"  Global clip deduplication: ENABLED (no clip reuse across timeline)")

    if oc.include_strategy_tracks:
        logger.info(f"  Strategy tracks: {', '.join(oc.strategy_tracks)}")
        # Handle vc as either object or dict
        if isinstance(vc, dict):
            req_diff = vc.get('require_different_source', True)
            min_time = vc.get('min_time_distance', 10.0)
            min_emb = vc.get('min_embedding_distance', 0.3)
        else:
            req_diff = getattr(vc, 'require_different_source', True)
            min_time = getattr(vc, 'min_time_distance', 10.0)
            min_emb = getattr(vc, 'min_embedding_distance', 0.3)
        logger.info(f"  Variety enforcement: different_source={req_diff}, "
                   f"min_time={min_time}s, min_emb_dist={min_emb}")

    # Build candidate embeddings lookup
    logger.info(f"Building candidate embeddings lookup for {len(video_segments)} segments...")
    candidate_embeddings = {}
    for seg, emb in zip(video_segments, video_embeddings):
        clip_id = strategy_matcher.get_clip_id(seg)
        candidate_embeddings[clip_id] = emb
    logger.info(f"Candidate embeddings lookup built ({len(candidate_embeddings)} entries)")

    # PRE-COMPUTE: B-roll segment list (computed once, used per voiceover segment)
    # B-roll segments use placeholder text, so their embeddings don't match voiceover semantically
    # We need to add them as candidates for the B-roll track (V8)
    all_broll_segments = [(seg, 1.0) for seg in video_segments if getattr(seg, 'is_broll', False)]
    broll_source_files = {seg.source_file for seg, _ in all_broll_segments}
    if all_broll_segments:
        logger.info(f"Pre-computed {len(all_broll_segments)} B-roll segments from {len(broll_source_files)} sources")

    progress = ProgressBar(len(voiceover_segments), "Matching")

    # US-85-005: Support within-stage resumption
    results: List[MatchResult] = list(prior_results) if prior_results else []
    if start_index > 0:
        logger.info(f"Resuming matching from segment {start_index} (skipping {start_index} already-matched segments)")
        # Fast-forward progress bar for already-matched segments
        progress.update(start_index, "resumed")

    # Calculate timeline start (first segment start time)
    timeline_start = voiceover_segments[0].start_time if voiceover_segments else 0.0

    logger.info(f"Starting matching loop with {len(video_segments)} video candidates...")

    # Validate embedding integrity before matching
    vo_validation = validate_embedding_integrity(voiceover_embeddings)
    vid_validation = validate_embedding_integrity(video_embeddings)

    if vo_validation.none_count > 0 or vo_validation.wrong_dimension_count > 0:
        log_error_with_context(
            logger, "MATCH-003",
            f"Voiceover embedding issues: {vo_validation.none_count} None, "
            f"{vo_validation.wrong_dimension_count} wrong dimension "
            f"(out of {vo_validation.total_count} total)"
        )
    if vid_validation.none_count > 0 or vid_validation.wrong_dimension_count > 0:
        log_error_with_context(
            logger, "MATCH-003",
            f"Video embedding issues: {vid_validation.none_count} None, "
            f"{vid_validation.wrong_dimension_count} wrong dimension "
            f"(out of {vid_validation.total_count} total)"
        )

    # Handle all-None embeddings: fall back to keyword-only matching
    if vo_validation.all_none or vid_validation.all_none:
        which = "voiceover" if vo_validation.all_none else "video"
        log_error_with_context(
            logger, "MATCH-003",
            f"All {which} embeddings are None — skipping embedding-based matching, "
            f"falling back to keyword-only matching"
        )
        # Pass video_segments so there's candidates to match against.
        # The EmbeddingSearch will return all segments as candidates when embeddings are empty,
        # allowing TieredMatcher to use keyword/text matching as fallback.
        embedding_search = EmbeddingSearch.from_matching_config(
            mc, [], video_segments, None
        )
    else:
        # Initialize embedding search with video segments and embeddings
        embedding_search = EmbeddingSearch.from_matching_config(
            mc, video_embeddings, video_segments, embedding_index
        )

    # Compute cross-chapter relevance matrix (US-72-009)
    relevance_matrix = None
    cg = getattr(mc, 'chapter_grouping', None)
    chapter_grouping_enabled = getattr(cg, 'enabled', True) if cg else False
    relevance_boost_weight = getattr(cg, 'relevance_boost_weight', 0.1) if cg else 0.1

    if chapter_grouping_enabled and location_chapters:
        # Build video chapter keyword lists from video segments grouped by chapter_index
        vid_chapter_keywords: Dict[int, set] = {}
        for seg in video_segments:
            ch_idx = getattr(seg, 'chapter_index', None)
            if ch_idx is not None and ch_idx >= 0:
                if ch_idx not in vid_chapter_keywords:
                    vid_chapter_keywords[ch_idx] = set()
                for kw in (seg.topics or []):
                    vid_chapter_keywords[ch_idx].add(kw)

        if vid_chapter_keywords:
            from ..chapter_detection.models import ChapterCandidate as CC
            # Convert location_chapters to ChapterCandidate if needed
            vo_chapters = []
            for ch in location_chapters:
                if isinstance(ch, CC):
                    vo_chapters.append(ch)
                elif isinstance(ch, dict):
                    vo_chapters.append(CC.from_dict(ch))
                elif hasattr(ch, 'topics'):
                    vo_chapters.append(CC(topics=getattr(ch, 'topics', [])))

            # Build video pseudo-chapters from segment topic groups
            max_vid_ch = max(vid_chapter_keywords.keys())
            vid_chapters = []
            for idx in range(max_vid_ch + 1):
                kws = list(vid_chapter_keywords.get(idx, set()))
                vid_chapters.append(CC(topics=kws))

            relevance_matrix = compute_relevance_matrix(vo_chapters, vid_chapters)
            if relevance_matrix:
                logger.info(
                    f"US-72-009 cross-chapter relevance matrix: "
                    f"{len(vo_chapters)}x{len(vid_chapters)} "
                    f"(boost_weight={relevance_boost_weight})"
                )

    # US-75-006: Pass relevance matrix to TieredMatcher for cross-chapter relevance boost
    if relevance_matrix:
        matcher.relevance_matrix = relevance_matrix

    # Build voiceover segment -> chapter index mapping for embedding boost
    # US-122-005: Handle multi_chapter_assignment_strategy for segments overlapping
    # multiple chapters or falling in gaps between chapters
    vo_segment_chapter_map: Dict[int, int] = {}
    if location_chapters:
        # Get the strategy from chapter_grouping config
        strategy = 'best_match'  # default
        if cg:
            strategy = getattr(cg, 'multi_chapter_assignment_strategy', 'best_match')

        # Build a list of (start, end, chapter_id) for all chapters
        chapter_ranges: List[Tuple[int, int, int]] = []
        for ch in location_chapters:
            ch_id = getattr(ch, 'chapter_id', None)
            if ch_id is None:
                continue
            start = getattr(ch, 'start_segment_idx', 0)
            end = getattr(ch, 'end_segment_idx', 0)
            chapter_ranges.append((start, end, ch_id))

        # Handle edge cases based on strategy
        if strategy == 'first':
            # Simply assign to the first chapter that contains each segment
            for seg_idx in range(len(voiceover_segments)):
                for start, end, ch_id in chapter_ranges:
                    if start <= seg_idx <= end:
                        vo_segment_chapter_map[seg_idx] = ch_id
                        break  # Stop at first match

        elif strategy == 'split':
            # For 'split', we handle gap cases by falling back to best_match
            # Actual segment splitting would require more complex logic
            # So we use best_match for gap segments
            for seg_idx in range(len(voiceover_segments)):
                overlapping = [(start, end, ch_id) for start, end, ch_id
                              in chapter_ranges if start <= seg_idx <= end]

                if len(overlapping) == 1:
                    # Single chapter overlap - use it directly
                    vo_segment_chapter_map[seg_idx] = overlapping[0][2]
                elif len(overlapping) > 1:
                    # Multiple chapters - use best_match (chapter with longest range)
                    best = max(overlapping, key=lambda x: x[1] - x[0])  # longest range wins
                    vo_segment_chapter_map[seg_idx] = best[2]
                else:
                    # Gap between chapters - use best_match (closest chapter)
                    # Find chapter with smallest gap
                    if chapter_ranges:
                        best_gap = min(chapter_ranges,
                                      key=lambda x: min(abs(x[1] - seg_idx), abs(x[0] - seg_idx)))
                        vo_segment_chapter_map[seg_idx] = best_gap[2]

        else:  # 'best_match' (default)
            # Assign to chapter with longest range when multiple overlap
            for seg_idx in range(len(voiceover_segments)):
                overlapping = [(start, end, ch_id) for start, end, ch_id
                              in chapter_ranges if start <= seg_idx <= end]

                if len(overlapping) == 1:
                    # Single chapter overlap - use it directly
                    vo_segment_chapter_map[seg_idx] = overlapping[0][2]
                elif len(overlapping) > 1:
                    # Multiple chapters - choose the one with longest range
                    best = max(overlapping, key=lambda x: x[1] - x[0])
                    vo_segment_chapter_map[seg_idx] = best[2]
                else:
                    # Segment in gap between chapters - leave unassigned (use -1)
                    # This allows downstream code to handle gaps specially
                    pass

    # US-75-010: Pass segment_chapter_map to TieredMatcher for per-segment chapter lookups
    if vo_segment_chapter_map:
        matcher.segment_chapter_map = vo_segment_chapter_map

    # US-77-002: Pass embedding lookup for semantic coherence scoring
    # Use len() check - numpy arrays can't be used directly in boolean context
    if video_embeddings is not None and len(video_embeddings) > 0 and video_segments:
        matcher.set_embedding_lookup(video_segments, video_embeddings)

    for i, (vo_seg, vo_emb) in enumerate(zip(voiceover_segments, voiceover_embeddings)):
        # US-85-005: Skip segments already matched during previous partial run
        if i < start_index:
            continue

        # Log first segment to confirm loop started
        if i == start_index:
            # Get voiceover segment ID if available
            seg_id = getattr(vo_seg, 'index', i)
            seg_start = getattr(vo_seg, 'start_time', 0)
            seg_end = getattr(vo_seg, 'end_time', 0)
            logger.info(f"[SEGMENT] id={seg_id} index={i} time={seg_start:.1f}-{seg_end:.1f}s \"{vo_seg.text[:50]}...\"")

        # Calculate current timeline position (relative to start)
        current_timeline_pos = vo_seg.start_time - timeline_start

        # Stage 1: Get candidates from embedding search for variety
        vo_chapter_idx = vo_segment_chapter_map.get(i, -1)
        all_candidates = embedding_search.search(
            vo_emb,
            relevance_matrix=relevance_matrix,
            voiceover_chapter_index=vo_chapter_idx,
            relevance_boost_weight=relevance_boost_weight,
        )

        # Add pre-computed B-roll segments to candidates (they may not be in top embedding matches)
        # Uses pre-computed all_broll_segments list (computed once outside loop)
        if all_broll_segments:
            # Add B-roll segments that aren't already in candidates
            candidate_sources = {seg.source_file for seg, _ in all_candidates}
            new_broll = [(seg, dist) for seg, dist in all_broll_segments
                        if seg.source_file not in candidate_sources]
            all_candidates.extend(new_broll)

        # US-162-007: Debug logging for candidate filtering - input count
        if i == 0:
            seg_start = getattr(vo_seg, 'start_time', 0)
            seg_end = getattr(vo_seg, 'end_time', 0)
            logger.info(f"[SEGMENT] id={seg_id} embedding search complete, {len(all_candidates)} candidates (time {seg_start:.1f}-{seg_end:.1f}s)")
            if all_broll_segments:
                logger.info(f"  Added {len(all_broll_segments)} B-roll segments to candidates")

        # US-141-009: Context-aware candidate pre-filtering
        # Pre-filter candidates by title/description/tags overlap before expensive matching
        context_prefilter_enabled = getattr(mc, 'context_prefilter_enabled', True)
        if context_prefilter_enabled and all_candidates:
            prefilter_count = len(all_candidates)
            context_threshold = getattr(mc, 'context_filter_threshold', 0.20)
            all_candidates = filter_by_context_relevance(
                candidates=all_candidates,
                vo_segment=vo_seg,
                threshold=context_threshold,
                video_metadata=video_metadata
            )
            postfilter_count = len(all_candidates)
            # US-162-007: Debug logging for candidate filtering (input -> output)
            logger.debug(f"[MATCH_DEBUG] seg_id={i} context_prefilter: {prefilter_count} -> {postfilter_count}")
            if i == 0 and prefilter_count != postfilter_count:
                logger.info(f"First segment: context prefilter removed {prefilter_count - postfilter_count} candidates")

        # Global clip deduplication: filter out clips already used anywhere in timeline
        if global_clip_tracker:
            pre_filter_count = len(all_candidates)
            deduped_candidates = [
                (seg, dist) for seg, dist in all_candidates
                if not global_clip_tracker.is_used(seg)
            ]
            dedup_count = len(deduped_candidates)

            # Keep hard dedup while pool is healthy, but relax if it would starve
            # candidate coverage for V1-V6 and force late-track collapse.
            # BUGFIX: relaxation must still exclude already-used clips to prevent
            # consecutive duplicate pairs. Using pre-dedup list defeats global dedup.
            relaxed = False
            if dedup_count < dedup_relax_threshold and dedup_count < pre_filter_count:
                relaxed = True
                # Still use deduped list even when relaxing - don't allow used clips
                # back into V1-V6 as this causes consecutive duplicates
                logger.warning(
                    f"[MATCH_DEBUG] seg_id={i} dedup relaxation triggered but still "
                    f"excluding used clips: {dedup_count} < {dedup_relax_threshold}"
                )
                all_candidates = deduped_candidates
            else:
                all_candidates = deduped_candidates

            postfilter_count = len(all_candidates)
            # US-162-007: Debug logging for candidate filtering (input -> output)
            if relaxed:
                logger.debug(
                    f"[MATCH_DEBUG] seg_id={i} global_dedup relaxed: "
                    f"{pre_filter_count} -> {dedup_count} (using {postfilter_count}, "
                    f"min_required={dedup_relax_threshold})"
                )
            else:
                logger.debug(f"[MATCH_DEBUG] seg_id={i} global_dedup: {pre_filter_count} -> {postfilter_count}")

            if i == 0 and pre_filter_count != dedup_count:
                logger.info(f"First segment: global dedup filtered {pre_filter_count - dedup_count} used clips")

        # US-135-006: Listicle boundary pre-filtering
        # Filter candidates based on listicle group boundaries before full matching
        listicle_boundary_config = mc.listicle_boundary if hasattr(mc, 'listicle_boundary') else None
        if listicle_boundary_config:
            lb_strictness = getattr(listicle_boundary_config, 'boundary_strictness', 'disabled')
            lb_min_overlap = getattr(listicle_boundary_config, 'min_topic_overlap', 0.2)
            lb_fallback = getattr(listicle_boundary_config, 'fallback_on_empty', True)

            if lb_strictness != 'disabled' and listicle_groups:
                pre_listicle_filter = len(all_candidates)
                filtered_by_listicle = filter_candidates_by_listicle(
                    candidates=all_candidates,
                    segment_idx=i,
                    listicle_groups=listicle_groups,
                    boundary_strictness=lb_strictness,
                    min_topic_overlap=lb_min_overlap,
                )

                # Apply fallback logic:
                # - If filtered list is not empty: use filtered list
                # - If filtered list is empty AND fallback enabled: use all candidates
                # - If filtered list is empty AND fallback disabled: use empty list
                if filtered_by_listicle:
                    all_candidates = filtered_by_listicle
                elif lb_fallback:
                    # Fallback: use all candidates when listicle filtering yields empty
                    logger.debug(f"Segment {i}: Listicle filter yielded no candidates, using fallback")
                # else: all_candidates stays as is (empty list)
                postfilter_count = len(all_candidates)
                # US-162-007: Debug logging for candidate filtering (input -> output)
                logger.debug(f"[MATCH_DEBUG] seg_id={i} listicle_filter: {pre_listicle_filter} -> {postfilter_count}")

                if i == 0 and pre_listicle_filter != len(all_candidates):
                    logger.info(f"First segment: listicle pre-filter removed {pre_listicle_filter - len(all_candidates)} candidates")

        # Apply timeline variety filtering for V1 (primary track)
        if variety_tracker:
            excluded_v1 = variety_tracker.get_excluded_sources("V1", current_timeline_pos)
            if excluded_v1:
                # Filter out excluded sources, but keep at least some candidates
                filtered_candidates = [(seg, dist) for seg, dist in all_candidates
                                       if seg.source_file not in excluded_v1]
                if len(filtered_candidates) >= mc.llm_rerank_candidates:
                    all_candidates = filtered_candidates
                else:
                    # Log that we had to relax the constraint
                    logger.debug(f"Segment {i}: Relaxed variety constraint (only {len(filtered_candidates)} candidates after filter)")

        # Stage 2: Send only top candidates to LLM for reranking
        llm_candidates = all_candidates[:mc.llm_rerank_candidates]
        # US-162-007: Debug logging for LLM candidate selection
        logger.debug(f"[MATCH_DEBUG] seg_id={i} LLM_candidates: {len(all_candidates)} -> {len(llm_candidates)} (top {mc.llm_rerank_candidates})")

        # Get context
        context_before = voiceover_segments[max(0, i - mc.context_window):i] if mc.context_window > 0 else None
        context_after = voiceover_segments[i+1:i+1+mc.context_window] if mc.context_window > 0 else None

        # Primary match (V1) - use only llm_rerank_candidates for LLM
        if i == 0:
            logger.info(f"First segment: calling LLM matcher with {len(llm_candidates)} candidates...")

        # Check if LLM reranking is disabled (embedding-only mode)
        if getattr(mc, 'llm_rerank', True):
            # Normal path: send candidates to LLM for reranking
            # US-164-012: Add error handling for matching failures
            try:
                result = matcher.match_segment(
                    vo_seg, llm_candidates, scenes,
                    context_before, context_after,
                    segment_idx=i  # Pass segment index for location chapter lookup
                )
            except Exception as e:
                log_error_with_context(
                    logger, "MATCH-001",
                    f"Matching failed for segment {i}: {e}",
                    segment_index=i,
                    segment_text=vo_seg.text[:50] if vo_seg.text else "N/A"
                )
                # Create a gap match as fallback
                result = MatchResult(
                    primary_match=create_gap_match(vo_seg, f"Matching failed: {e}"),
                    has_gap=True,
                    gap_reason=f"Matching error: {e}"
                )

            # Guard: wrap bare Match in MatchResult if match_segment returned wrong type
            if not hasattr(result, 'primary_match') and hasattr(result, 'confidence'):
                result = MatchResult(primary_match=result)

            if i == 0:
                logger.info(f"First segment: LLM match complete, confidence={result.primary_match.confidence:.2f}")
        else:
            # Embedding-only mode: use best candidate from stage 1 directly
            best_candidate = all_candidates[0][0] if all_candidates else None
            if not best_candidate:
                result = MatchResult(
                    primary_match=create_gap_match(vo_seg, "No embedding candidates found"),
                    has_gap=True,
                    gap_reason="No candidates"
                )
            else:
                # Build MatchResult from embedding candidate
                # Use video_segment=best_candidate (SRTSegment) so downstream code that
                # expects .video_segment.source_file works correctly
                top_similarity = all_candidates[0][1]
                match = Match(
                    segment_index=i,
                    video_file=best_candidate.source_file,
                    video_start=best_candidate.start_time,
                    video_end=best_candidate.end_time,
                    confidence=float(top_similarity),
                    strategy="embedding_only",
                    reason=f"Embedding match (similarity={top_similarity:.3f}), llm_rerank=False"
                )
                # Attach video_segment for compatibility with code expecting .video_segment
                match.video_segment = best_candidate
                match.voiceover_segment = vo_seg

                # Generate alternatives for V2-V3 tracks using the same AlternativeSelector as LLM path
                alternatives = matcher.alt_selector.get_alternatives(
                    all_candidates[1:4], scenes, best_candidate, matcher._get_scene_for_segment
                )

                # DEBUG: log alternatives and used_segments
                if i == 0:
                    logger.info(f"[DEBUG] alternatives={len(alternatives)}, V1={best_candidate.source_file}:{best_candidate.start_time}")
                    for ai, alt in enumerate(alternatives):
                        vs = alt.video_segment
                        logger.info(f"[DEBUG]   alt[{ai}]={vs.source_file}:{vs.start_time}")

                # Generate secondary matches for V4-V6 tracks
                # In embedding-only mode, take the next best candidates after V1 directly as secondary matches
                # Only skip exact (source, start_time) duplicates - allow same source with different segment
                # This allows V4-V6 when candidate pool is limited to same sources as V1-V3
                from ..utils import AlternativeMatch
                secondary_matches = []
                used_segments = {(best_candidate.source_file, best_candidate.start_time)}
                for alt in alternatives:
                    vs = alt.video_segment
                    used_segments.add((vs.source_file, vs.start_time))

                if i == 0:
                    logger.info(f"[DEBUG] used_segments after alternatives: {used_segments}")
                    logger.info(f"[DEBUG] all_candidates[1:] count={len(all_candidates[1:])}")

                # Start from index 1 (after V1), skip only exact segment duplicates
                for seg, sim in all_candidates[1:]:
                    if len(secondary_matches) >= 3:
                        break
                    seg_key = (seg.source_file, seg.start_time)
                    if seg_key in used_segments:
                        continue
                    scene = matcher._get_scene_for_segment(seg, scenes) if scenes else None
                    position = len(secondary_matches)
                    label = "Secondary Primary" if position == 0 else f"Secondary Alt {position}"
                    secondary_matches.append(AlternativeMatch(
                        video_segment=seg,
                        video_scene=scene,
                        confidence=sim,
                        reasoning=f"{label} (source: {seg.source_file})"
                    ))
                    used_segments.add(seg_key)

                result = MatchResult(primary_match=match, alternatives=alternatives, secondary_matches=secondary_matches)
                if i == 0:
                    logger.info(f"First segment: embedding-only match, confidence={top_similarity:.2f}, alternatives={len(alternatives)}, secondary={len(secondary_matches)}")

        # Record V1 usage for timeline variety and global clip tracker
        if result.primary_match:
            primary = result.primary_match
            # Safe access: Match uses video_file directly, LLM-wrapped Match has video_segment
            source_file = getattr(getattr(primary, 'video_segment', None), 'source_file', None) or getattr(primary, 'video_file', None)
            if variety_tracker:
                variety_tracker.record_usage(
                    "V1",
                    source_file,
                    current_timeline_pos
                )
            if global_clip_tracker:
                video_seg = getattr(primary, 'video_segment', None) or best_candidate
                global_clip_tracker.record_usage(
                    video_seg, "V1", i
                )

        # Record V2, V3 (alternatives) usage
        # US-162-007: Debug logging for alternatives considered per segment
        if result.alternatives:
            alt_conf_str = ", ".join(
                f"V{alt_idx}:{getattr(alt, 'confidence', 0):.2f}"
                for alt_idx, alt in enumerate(result.alternatives, start=2)
            )
            logger.debug(f"[MATCH_DEBUG] seg_id={i} alternatives: {len(result.alternatives)} - {alt_conf_str}")
            for alt_idx, alt in enumerate(result.alternatives, start=2):
                alt_source = getattr(getattr(alt, 'video_segment', None), 'source_file', None) or getattr(alt, 'video_file', None)
                if variety_tracker:
                    variety_tracker.record_usage(
                        f"V{alt_idx}",
                        alt_source,
                        current_timeline_pos
                    )
                if global_clip_tracker:
                    alt_vs = getattr(alt, 'video_segment', None)
                    global_clip_tracker.record_usage(
                        alt_vs, f"V{alt_idx}", i
                    )

        # Strategy matches (V4-V10) - use all embedding candidates for variety
        if oc.include_strategy_tracks:
            # Get alternative segments (V2-V3) - safe access for both embedding-only and LLM paths
            alt_segments = []
            for alt in result.alternatives:
                vs = getattr(alt, 'video_segment', None)
                if vs is not None:
                    alt_segments.append(vs)

            # Apply timeline variety filtering for strategy candidates
            strategy_candidates = all_candidates
            if variety_tracker:
                # For strategy tracks, exclude sources used in V1-V3 AND timeline-excluded sources
                # Use a combined track "V_strategy" for variety tracking
                excluded_strategy = variety_tracker.get_excluded_sources("V_strategy", current_timeline_pos)
                if excluded_strategy:
                    filtered_strategy = [(seg, dist) for seg, dist in all_candidates
                                        if seg.source_file not in excluded_strategy]
                    if len(filtered_strategy) >= 5:  # Need at least some candidates
                        strategy_candidates = filtered_strategy

            # Compute strategy matches with variety enforcement
            # V7-V10 don't use global clip tracker - they can reuse clips from V1-V3
            # This gives more options for strategy tracks without exhausting the candidate pool
            primary_vs = getattr(result.primary_match, 'video_segment', None)
            if primary_vs is None:
                primary_vs = best_candidate  # embedding-only Match has no video_segment
            strategy_matches = strategy_matcher.get_strategy_matches(
                vo_segment=vo_seg,
                all_candidates=strategy_candidates,  # Use filtered candidates
                primary_match=primary_vs,
                secondary_matches=alt_segments,
                vo_embedding=vo_emb,
                candidate_embeddings=candidate_embeddings,
                segment_index=i
            )

            result.strategy_matches = strategy_matches

            # Record strategy track usage (variety tracker only - NOT global clip tracker)
            # V7+ can reuse clips from V1-V3, so we don't add them to global tracker
            if variety_tracker:
                for sm in strategy_matches:
                    sm_src = getattr(getattr(sm, 'video_segment', None), 'source_file', '') or ''
                    variety_tracker.record_usage(
                        "V_strategy",
                        sm_src,
                        current_timeline_pos
                    )
            # NOTE: Intentionally NOT recording V7+ in global_clip_tracker
            # This allows strategy tracks to reuse clips without exhausting the pool

            # Compute secondary matches (V4-V6) using diversity scoring
            # In embedding-only mode, skip this - secondary_matches already set above
            # In LLM mode, this overrides with strategy_matcher strict source enforcement
            if getattr(mc, 'llm_rerank', True):
                secondary_matches = strategy_matcher.get_secondary_matches_diversity(
                    vo_segment=vo_seg,
                    all_candidates=strategy_candidates,
                    primary_match=primary_vs,
                    secondary_matches=alt_segments,
                    vo_embedding=vo_emb,
                    candidate_embeddings=candidate_embeddings
                )
                result.secondary_matches = secondary_matches

            # Record V4-V6 usage for timeline variety only (NOT global clip tracker)
            # This allows secondary tracks to reuse clips without exhausting the pool
            if result.secondary_matches:
                for sec_idx, sec_match in enumerate(secondary_matches, start=4):
                    if variety_tracker:
                        sec_src = getattr(getattr(sec_match, 'video_segment', None), 'source_file', '') or ''
                        variety_tracker.record_usage(
                            f"V{sec_idx}",
                            sec_src,
                            current_timeline_pos
                        )
                    # NOTE: Intentionally NOT recording V4-V6 in global_clip_tracker

        results.append(result)

        # US-85-005: Invoke progress callback for intermediate checkpointing
        if progress_callback is not None:
            progress_callback(i, results)

        # Progress with strategy count
        strat_count = len(result.strategy_matches) if result.strategy_matches else 0
        progress.update(1, f"conf: {result.primary_match.confidence:.2f}, strat: {strat_count}")

    progress.close()

    # Review low-confidence matches with local LLM
    if config.matching.use_local_for_review and matcher.local_provider:
        results = matcher.review_with_local_llm(results)

    # US-77-007: Enforce minimum source diversity per chapter
    results = matcher.enforce_chapter_source_diversity(results)

    # Normalize: ensure all results are MatchResult (bare Match objects from
    # checkpoint restore or fallback paths cause AttributeError downstream)
    for idx, r in enumerate(results):
        if r and not hasattr(r, 'primary_match') and hasattr(r, 'confidence'):
            results[idx] = MatchResult(primary_match=r)

    # US-77-008: Log scoring adjustment audit summary
    # US-134-011: Include context cache statistics
    from .tiered_matcher import compute_scoring_audit_summary, log_scoring_audit_summary
    context_cache_stats = None
    if hasattr(matcher, 'llm_reranker') and matcher.llm_reranker:
        context_cache_stats = matcher.llm_reranker.get_context_cache_stats()
    scoring_audit = compute_scoring_audit_summary(results, context_cache_stats)
    log_scoring_audit_summary(scoring_audit)

    # Report gaps
    gaps = [r for r in results if r.has_gap]
    if gaps:
        logger.warning(f"Found {len(gaps)} footage gaps (low confidence matches)")
        for gap in gaps[:5]:  # Show first 5
            pm = gap.primary_match
            vo_seg = getattr(pm, 'voiceover_segment', None)
            vo_text = getattr(vo_seg, 'text', '')[:50] if vo_seg else 'N/A'
            logger.warning(f"  - \"{vo_text}...\" ({gap.gap_reason})")

    # Report strategy match stats
    if oc.include_strategy_tracks:
        for strategy in oc.strategy_tracks:
            count = sum(1 for r in results for sm in r.strategy_matches if sm.strategy == strategy)
            logger.info(f"  {strategy}: {count}/{len(results)} segments matched")

    # Report timeline variety stats
    if variety_tracker:
        stats = variety_tracker.get_stats()
        logger.info(f"  Timeline variety enforcement:")
        for track, track_stats in stats.items():
            if track_stats["total_clips"] > 0:
                unique_pct = track_stats["unique_sources"] / track_stats["total_clips"] * 100
                logger.info(f"    {track}: {track_stats['unique_sources']} unique sources across {track_stats['total_clips']} clips ({unique_pct:.0f}% variety)")
                if track_stats["top_sources"]:
                    top = track_stats["top_sources"][0]
                    logger.info(f"      Most used: {top[0]} ({top[1]} times)")

    # Report global clip deduplication stats
    if global_clip_tracker:
        global_stats = global_clip_tracker.get_stats()
        logger.info(f"  Global clip deduplication:")
        logger.info(f"    Total clips placed: {global_stats['total_clips_used']}")
        logger.info(f"    All clips unique: YES (hard block enforced)")

    # === VARIETY METRICS DASHBOARD ===
    logger.info("=" * 60)
    logger.info("VARIETY METRICS DASHBOARD")
    logger.info("=" * 60)

    # Track coverage stats
    v1_matched = sum(1 for r in results if _is_usable_primary_match(r, mc.min_confidence))
    # US-164-012: Log segments filtered by min_confidence threshold
    below_threshold = sum(
        1 for r in results
        if _is_usable_primary_match(r) and r.primary_match.confidence < mc.min_confidence
    )
    if below_threshold > 0:
        logger.info(f"  [MATCH_CONFIDENCE] {below_threshold} segments filtered by min_confidence threshold ({mc.min_confidence})")
    v2_matched = sum(1 for r in results if r.alternatives and len(r.alternatives) >= 1)
    v3_matched = sum(1 for r in results if r.alternatives and len(r.alternatives) >= 2)
    v4_matched = sum(1 for r in results if r.secondary_matches and len(r.secondary_matches) >= 1)
    v5_matched = sum(1 for r in results if r.secondary_matches and len(r.secondary_matches) >= 2)
    v6_matched = sum(1 for r in results if r.secondary_matches and len(r.secondary_matches) >= 3)
    v7_matched = sum(1 for r in results if r.strategy_matches and len(r.strategy_matches) >= 1)

    total_segs = len(results)
    if total_segs > 0:
        logger.info(f"  Track Coverage:")
        logger.info(f"    V1 (Primary):    {v1_matched:4d}/{total_segs} ({v1_matched/total_segs*100:.1f}%)")
        logger.info(f"    V2 (Alt 1):      {v2_matched:4d}/{total_segs} ({v2_matched/total_segs*100:.1f}%)")
        logger.info(f"    V3 (Alt 2):      {v3_matched:4d}/{total_segs} ({v3_matched/total_segs*100:.1f}%)")
        logger.info(f"    V4 (Sec Pri):    {v4_matched:4d}/{total_segs} ({v4_matched/total_segs*100:.1f}%)")
        logger.info(f"    V5 (Sec Alt 1):  {v5_matched:4d}/{total_segs} ({v5_matched/total_segs*100:.1f}%)")
        logger.info(f"    V6 (Sec Alt 2):  {v6_matched:4d}/{total_segs} ({v6_matched/total_segs*100:.1f}%)")
        logger.info(f"    V7 (Strategy):   {v7_matched:4d}/{total_segs} ({v7_matched/total_segs*100:.1f}%)")
    else:
        logger.info(f"  Track Coverage: No segments to match")

    # Source concentration analysis
    v1_sources: Dict[str, int] = defaultdict(int)
    for r in results:
        if not _is_usable_primary_match(r, mc.min_confidence):
            continue
        pm = r.primary_match
        has_vs = hasattr(pm, 'video_segment')
        pm_video_file = getattr(pm, 'video_file', 'NO_VIDEO_FILE')
        src_file = getattr(getattr(pm, 'video_segment', None), 'source_file', None) or pm_video_file
        src_name = Path(src_file).stem[:20]
        v1_sources[src_name] += 1

    if v1_sources:
        sorted_sources = sorted(v1_sources.items(), key=lambda x: -x[1])
        top3_count = sum(count for _, count in sorted_sources[:3])
        top3_pct = top3_count / total_segs * 100 if total_segs > 0 else 0

        logger.info(f"  Source Concentration (V1):")
        logger.info(f"    Unique sources: {len(v1_sources)}")
        logger.info(f"    Top 3 sources: {top3_pct:.1f}% of clips")
        for src, count in sorted_sources[:3]:
            logger.info(f"      {src}: {count} clips ({count/total_segs*100:.1f}%)")

    # Gap analysis
    if total_segs > 0:
        low_alt_segments = sum(1 for r in results if len(r.alternatives or []) < 2)
        no_secondary = sum(1 for r in results if not r.secondary_matches)
        logger.info(f"  Gap Analysis:")
        logger.info(f"    Segments with <2 alternatives: {low_alt_segments}/{total_segs} ({low_alt_segments/total_segs*100:.1f}%)")
        logger.info(f"    Segments with no V4-V6:       {no_secondary}/{total_segs} ({no_secondary/total_segs*100:.1f}%)")

    # Confidence distribution
    confidences = [r.primary_match.confidence for r in results if _is_usable_primary_match(r)]
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        # US-164-012: Add median and std to confidence distribution
        sorted_conf = sorted(confidences)
        n = len(sorted_conf)
        median_conf = sorted_conf[n // 2] if n % 2 == 1 else (sorted_conf[n // 2 - 1] + sorted_conf[n // 2]) / 2
        variance = sum((c - avg_conf) ** 2 for c in confidences) / n
        std_conf = variance ** 0.5

        high_conf = sum(1 for c in confidences if c >= 0.85)
        med_conf = sum(1 for c in confidences if 0.5 <= c < 0.85)
        low_conf = sum(1 for c in confidences if c < 0.5)
        logger.info(f"  Confidence Distribution (V1):")
        logger.info(f"    Mean: {avg_conf:.3f}, Median: {median_conf:.3f}, Std: {std_conf:.3f}")
        logger.info(f"    High (≥0.85): {high_conf} | Medium (0.5-0.85): {med_conf} | Low (<0.5): {low_conf}")

    logger.info("=" * 60)

    # === SOURCE DIVERSITY METRICS (US-53-005) ===
    from .metrics import compute_diversity_metrics, log_diversity_metrics
    diversity_report = compute_diversity_metrics(results)
    log_diversity_metrics(diversity_report)

    # Analyze low confidence segments
    analyze_low_confidence_segments(results)

    # Report very low confidence segments with breakdown (US-77-012)
    from .tiered_matcher import report_low_confidence_segments
    report_low_confidence_segments(results)

    # Log cache statistics for performance analysis
    try:
        from .similarity_cache import log_all_cache_stats
        log_all_cache_stats()
    except ImportError:
        pass

    return results


@dataclass
class LowConfidencePattern:
    """Pattern detected in low confidence segments."""
    pattern_type: str  # e.g., "short_voiceover", "abstract_content", "missing_keywords"
    count: int
    segment_indices: List[int]
    description: str


@dataclass
class LowConfidenceAnalysis:
    """Analysis result for low confidence segments."""
    low_confidence_count: int
    total_segments: int
    patterns: List[LowConfidencePattern]
    suggestions: List[str]
    avg_low_confidence: float
    threshold: float


def analyze_low_confidence_segments(
    results: List[MatchResult],
    threshold: float = 0.6
) -> LowConfidenceAnalysis:
    """
    Analyze segments with confidence below threshold to identify common patterns.

    Identifies patterns such as:
    - Short voiceover segments (< 20 characters)
    - Abstract content (lacking concrete nouns/keywords)
    - Missing keywords (no keywords extracted)
    - High confidence variance (uncertain matches)

    Logs analysis and suggestions for improvement.

    Args:
        results: List of MatchResult objects from matching
        threshold: Confidence threshold (default 0.6)

    Returns:
        LowConfidenceAnalysis with patterns and suggestions
    """
    if not results:
        return LowConfidenceAnalysis(
            low_confidence_count=0,
            total_segments=0,
            patterns=[],
            suggestions=[],
            avg_low_confidence=0.0,
            threshold=threshold
        )

    # Identify low confidence segments
    low_conf_segments = []
    for i, r in enumerate(results):
        if r.primary_match and r.primary_match.confidence < threshold:
            low_conf_segments.append((i, r))

    if not low_conf_segments:
        logger.info(f"No low confidence segments (threshold: {threshold})")
        return LowConfidenceAnalysis(
            low_confidence_count=0,
            total_segments=len(results),
            patterns=[],
            suggestions=[],
            avg_low_confidence=0.0,
            threshold=threshold
        )

    # Calculate average confidence of low segments
    avg_low_conf = sum(r.primary_match.confidence for _, r in low_conf_segments) / len(low_conf_segments)

    # Detect patterns
    patterns = []
    suggestions = []

    # Pattern 1: Short voiceover (< 20 characters)
    short_vo_indices = []
    for i, r in low_conf_segments:
        pm = r.primary_match
        vo_seg = getattr(pm, 'voiceover_segment', None)
        if vo_seg and len(vo_seg.text.strip()) < 20:
            short_vo_indices.append(i)

    if short_vo_indices:
        patterns.append(LowConfidencePattern(
            pattern_type="short_voiceover",
            count=len(short_vo_indices),
            segment_indices=short_vo_indices,
            description="Voiceover text too short (< 20 chars) - insufficient context for matching"
        ))
        suggestions.append("Consider merging short voiceover segments or adding more descriptive text")

    # Pattern 2: Missing keywords
    no_keywords_indices = []
    for i, r in low_conf_segments:
        pm = r.primary_match
        vo_seg = getattr(pm, 'voiceover_segment', None)
        if not vo_seg:
            no_keywords_indices.append(i)
            continue
        keywords = getattr(vo_seg, 'keywords', []) or []
        if len(keywords) == 0:
            no_keywords_indices.append(i)

    if no_keywords_indices:
        patterns.append(LowConfidencePattern(
            pattern_type="missing_keywords",
            count=len(no_keywords_indices),
            segment_indices=no_keywords_indices,
            description="No keywords extracted from voiceover - semantic matching limited"
        ))
        suggestions.append("Run keyword extraction stage or manually add keywords to voiceover segments")

    # Pattern 3: Abstract content (few matched keywords with video)
    abstract_indices = []
    for i, r in low_conf_segments:
        matched_kws = r.matched_keywords or []
        pm = r.primary_match
        vo_seg = getattr(pm, 'voiceover_segment', None)
        if not vo_seg:
            continue
        # Check if voiceover has abstract words without concrete nouns
        text_lower = vo_seg.text.lower()
        abstract_words = ["thing", "stuff", "something", "everything", "nothing", "way", "kind", "sort"]
        has_abstract = any(word in text_lower for word in abstract_words)
        few_matches = len(matched_kws) < 2
        if has_abstract and few_matches:
            abstract_indices.append(i)

    if abstract_indices:
        patterns.append(LowConfidencePattern(
            pattern_type="abstract_content",
            count=len(abstract_indices),
            segment_indices=abstract_indices,
            description="Voiceover contains abstract language with few concrete keywords"
        ))
        suggestions.append("Add more specific terminology or entity names to improve matching")

    # Pattern 4: High confidence variance (uncertain matches)
    high_variance_indices = []
    for i, r in low_conf_segments:
        variance = r.confidence_variance
        if variance > 0.15:
            high_variance_indices.append(i)

    if high_variance_indices:
        patterns.append(LowConfidencePattern(
            pattern_type="high_variance",
            count=len(high_variance_indices),
            segment_indices=high_variance_indices,
            description="High variance among candidate scores - ambiguous matches"
        ))
        suggestions.append("Consider adding more specific video footage or adjusting matching parameters")

    # Pattern 5: No matched keywords between VO and video
    no_keyword_match_indices = []
    for i, r in low_conf_segments:
        matched_kws = r.matched_keywords or []
        if len(matched_kws) == 0:
            # Don't double-count if already in missing_keywords
            if i not in no_keywords_indices:
                no_keyword_match_indices.append(i)

    if no_keyword_match_indices:
        patterns.append(LowConfidencePattern(
            pattern_type="no_keyword_overlap",
            count=len(no_keyword_match_indices),
            segment_indices=no_keyword_match_indices,
            description="No keyword overlap between voiceover and matched video"
        ))
        suggestions.append("Ensure video candidates have relevant transcripts or metadata")

    # Pattern 6: Source concentration (multiple low-conf segments use same video)
    source_counts: Dict[str, List[int]] = {}
    for i, r in low_conf_segments:
        pm = r.primary_match
        src_file = getattr(getattr(pm, 'video_segment', None), 'source_file', None) or getattr(pm, 'video_file', None)
        if src_file:
            source_counts.setdefault(src_file, []).append(i)

    concentrated_sources = {src: indices for src, indices in source_counts.items() if len(indices) >= 3}
    if concentrated_sources:
        # Collect all segment indices affected by source concentration
        concentrated_indices = []
        for indices in concentrated_sources.values():
            concentrated_indices.extend(indices)
        concentrated_indices = sorted(set(concentrated_indices))

        source_list = ", ".join(f"{src} ({len(idx)}x)" for src, idx in concentrated_sources.items())
        patterns.append(LowConfidencePattern(
            pattern_type="source_concentration",
            count=len(concentrated_indices),
            segment_indices=concentrated_indices,
            description=f"Multiple low-confidence segments matched to same source: {source_list}"
        ))
        suggestions.append("Expand video pool with more diverse sources to reduce reliance on a single video")

    # Log analysis
    logger.info("=" * 60)
    logger.info("LOW CONFIDENCE SEGMENT ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"  Threshold: {threshold}")
    logger.info(f"  Low confidence segments: {len(low_conf_segments)}/{len(results)} ({len(low_conf_segments)/len(results)*100:.1f}%)")
    logger.info(f"  Average confidence (low segments): {avg_low_conf:.3f}")

    if patterns:
        logger.info(f"  Patterns detected:")
        for p in patterns:
            logger.info(f"    {p.pattern_type}: {p.count} segments")
            logger.info(f"      {p.description}")

    if suggestions:
        logger.info(f"  Suggestions for improvement:")
        for i, suggestion in enumerate(suggestions, 1):
            logger.info(f"    {i}. {suggestion}")

    # Log sample of low confidence segments
    if low_conf_segments:
        logger.info(f"  Sample low confidence segments (first 5):")
        for idx, r in low_conf_segments[:5]:
            pm = r.primary_match
            vo_seg = getattr(pm, 'voiceover_segment', None)
            vo_text = getattr(vo_seg, 'text', '')[:50] if vo_seg else 'N/A'
            conf = getattr(pm, 'confidence', 0.0)
            logger.info(f"    Segment {idx}: \"{vo_text}...\" (conf: {conf:.2f})")

    logger.info("=" * 60)

    return LowConfidenceAnalysis(
        low_confidence_count=len(low_conf_segments),
        total_segments=len(results),
        patterns=patterns,
        suggestions=suggestions,
        avg_low_confidence=avg_low_conf,
        threshold=threshold
    )
