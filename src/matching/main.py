"""
Main matching orchestration.

Migrated from matching.py (lines 2484-2872, ~390 lines).
Provides the public match_all_segments() API for voiceover-to-video matching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Dict, Any
from collections import defaultdict
from pathlib import Path
import logging

from .tracking import TimelineVarietyTracker, GlobalClipTracker
from .strategies import StrategyMatcher
from ..utils import SRTSegment, MatchResult, ProgressBar
from ..embeddings import find_top_k_similar

if TYPE_CHECKING:
    from ..config import Config
    from ..utils import CacheManager, SceneInfo, VideoTopics
    from ..topic_extraction import LocationChapter
    from ..location_service import GeoLocation

logger = logging.getLogger(__name__)


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
    video_locations: Optional[Dict[str, 'GeoLocation']] = None
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

    Returns:
        List of MatchResult objects, one per voiceover segment
    """
    # Import TieredMatcher here to avoid circular import
    from .tiered_matcher import TieredMatcher

    matcher = TieredMatcher(config, cache, video_topics=video_topics)

    # Set up location-aware matching if provided
    if location_chapters:
        matcher.set_location_chapters(location_chapters)
    if video_locations:
        matcher.set_video_locations(video_locations)
    strategy_matcher = StrategyMatcher(config, scenes)

    # Store face preference for use during matching
    matcher.face_preference = face_preference

    mc = config.matching
    oc = config.output
    vc = oc.variety

    logger.info(f"Matching {len(voiceover_segments)} voiceover segments...")
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

    progress = ProgressBar(len(voiceover_segments), "Matching")

    results = []

    # Calculate timeline start (first segment start time)
    timeline_start = voiceover_segments[0].start_time if voiceover_segments else 0.0

    logger.info(f"Starting matching loop with {len(video_segments)} video candidates...")

    for i, (vo_seg, vo_emb) in enumerate(zip(voiceover_segments, voiceover_embeddings)):
        # Log first segment to confirm loop started
        if i == 0:
            logger.info(f"Processing first segment: \"{vo_seg.text[:50]}...\"")

        # Calculate current timeline position (relative to start)
        current_timeline_pos = vo_seg.start_time - timeline_start

        # Stage 1: Get more candidates from embedding search for variety
        num_embedding_candidates = max(mc.embedding_candidates, 20)
        distances, indices = find_top_k_similar(vo_emb, video_embeddings, num_embedding_candidates, index=embedding_index)
        # Filter out-of-bounds indices (can occur if videos were filtered after embedding)
        valid_indices = [j for j, idx in enumerate(indices) if idx < len(video_segments)]
        all_candidates = [(video_segments[indices[j]], distances[j]) for j in valid_indices]

        # Add ALL B-roll segments to candidates (they may not be in top embedding matches)
        # B-roll segments use placeholder text, so their embeddings don't match voiceover semantically
        broll_segments = [(seg, 1.0) for seg in video_segments if getattr(seg, 'is_broll', False)]
        if broll_segments:
            # Add B-roll segments that aren't already in candidates
            candidate_sources = {seg.source_file for seg, _ in all_candidates}
            new_broll = [(seg, dist) for seg, dist in broll_segments
                        if seg.source_file not in candidate_sources]
            all_candidates.extend(new_broll)

        if i == 0:
            logger.info(f"First segment: embedding search complete, {len(all_candidates)} candidates")
            if broll_segments:
                logger.info(f"  Added {len(broll_segments)} B-roll segments to candidates")

        # Filter out caption-only segments BEFORE LLM reranking
        # Caption-only segments help with embedding search but shouldn't be final candidates
        pre_filter_caption_count = len(all_candidates)
        all_candidates = [
            (seg, dist) for seg, dist in all_candidates
            if not getattr(seg, 'caption_only', False)
        ]
        caption_filtered = pre_filter_caption_count - len(all_candidates)
        if i == 0 and caption_filtered > 0:
            logger.info(f"First segment: filtered {caption_filtered} caption-only segments after embedding search")

        # Global clip deduplication: filter out clips already used anywhere in timeline
        if global_clip_tracker:
            pre_filter_count = len(all_candidates)
            all_candidates = [
                (seg, dist) for seg, dist in all_candidates
                if not global_clip_tracker.is_used(seg)
            ]
            if i == 0 and pre_filter_count != len(all_candidates):
                logger.info(f"First segment: global dedup filtered {pre_filter_count - len(all_candidates)} used clips")

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

        # Get context
        context_before = voiceover_segments[max(0, i - mc.context_window):i] if mc.context_window > 0 else None
        context_after = voiceover_segments[i+1:i+1+mc.context_window] if mc.context_window > 0 else None

        # Primary match (V1) - use only llm_rerank_candidates for LLM
        if i == 0:
            logger.info(f"First segment: calling LLM matcher with {len(llm_candidates)} candidates...")
        result = matcher.match_segment(
            vo_seg, llm_candidates, scenes,
            context_before, context_after,
            segment_idx=i  # Pass segment index for location chapter lookup
        )
        if i == 0:
            logger.info(f"First segment: LLM match complete, confidence={result.primary_match.confidence:.2f}")

        # Record V1 usage for timeline variety and global clip tracker
        if result.primary_match:
            if variety_tracker:
                variety_tracker.record_usage(
                    "V1",
                    result.primary_match.video_segment.source_file,
                    current_timeline_pos
                )
            if global_clip_tracker:
                global_clip_tracker.record_usage(
                    result.primary_match.video_segment, "V1", i
                )

        # Record V2, V3 (alternatives) usage
        if result.alternatives:
            for alt_idx, alt in enumerate(result.alternatives, start=2):
                if variety_tracker:
                    variety_tracker.record_usage(
                        f"V{alt_idx}",
                        alt.video_segment.source_file,
                        current_timeline_pos
                    )
                if global_clip_tracker:
                    global_clip_tracker.record_usage(
                        alt.video_segment, f"V{alt_idx}", i
                    )

        # Strategy matches (V4-V10) - use all embedding candidates for variety
        if oc.include_strategy_tracks:
            # Get alternative segments (V2-V3)
            alt_segments = [alt.video_segment for alt in result.alternatives]

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
            strategy_matches = strategy_matcher.get_strategy_matches(
                vo_segment=vo_seg,
                all_candidates=strategy_candidates,  # Use filtered candidates
                primary_match=result.primary_match.video_segment,
                alternatives=alt_segments,
                vo_embedding=vo_emb,
                candidate_embeddings=candidate_embeddings,
                segment_index=i,
                global_used_clips=None  # V7+ can reuse clips
            )

            result.strategy_matches = strategy_matches

            # Record strategy track usage (variety tracker only - NOT global clip tracker)
            # V7+ can reuse clips from V1-V3, so we don't add them to global tracker
            if variety_tracker:
                for sm in strategy_matches:
                    variety_tracker.record_usage(
                        "V_strategy",
                        sm.video_segment.source_file,
                        current_timeline_pos
                    )
            # NOTE: Intentionally NOT recording V7+ in global_clip_tracker
            # This allows strategy tracks to reuse clips without exhausting the pool

            # Compute secondary matches (V4-V6) using diversity scoring
            # This overrides the secondary_matches from match_segment with strict source enforcement
            # V4-V6 don't use global clip tracker - they can reuse clips from V1-V3
            secondary_matches = strategy_matcher.get_secondary_matches_diversity(
                vo_segment=vo_seg,
                all_candidates=strategy_candidates,
                primary_match=result.primary_match.video_segment,
                alternatives=alt_segments,
                vo_embedding=vo_emb,
                candidate_embeddings=candidate_embeddings,
                global_used_clips=None  # V4-V6 can reuse clips
            )
            result.secondary_matches = secondary_matches

            # Record V4-V6 usage for timeline variety only (NOT global clip tracker)
            # This allows secondary tracks to reuse clips without exhausting the pool
            if secondary_matches:
                for sec_idx, sec_match in enumerate(secondary_matches, start=4):
                    if variety_tracker:
                        variety_tracker.record_usage(
                            f"V{sec_idx}",
                            sec_match.video_segment.source_file,
                            current_timeline_pos
                        )
                    # NOTE: Intentionally NOT recording V4-V6 in global_clip_tracker

        results.append(result)

        # Progress with strategy count
        strat_count = len(result.strategy_matches) if result.strategy_matches else 0
        progress.update(1, f"conf: {result.primary_match.confidence:.2f}, strat: {strat_count}")

    progress.close()

    # Review low-confidence matches with local LLM
    if config.matching.use_local_for_review and matcher.local_provider:
        results = matcher.review_with_local_llm(results)

    # Report gaps
    gaps = [r for r in results if r.has_gap]
    if gaps:
        logger.warning(f"Found {len(gaps)} footage gaps (low confidence matches)")
        for gap in gaps[:5]:  # Show first 5
            logger.warning(f"  - \"{gap.primary_match.voiceover_segment.text[:50]}...\" ({gap.gap_reason})")

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
    v1_matched = sum(1 for r in results if r.primary_match and r.primary_match.confidence >= mc.min_confidence)
    v2_matched = sum(1 for r in results if r.alternatives and len(r.alternatives) >= 1)
    v3_matched = sum(1 for r in results if r.alternatives and len(r.alternatives) >= 2)
    v4_matched = sum(1 for r in results if r.secondary_matches and len(r.secondary_matches) >= 1)
    v5_matched = sum(1 for r in results if r.secondary_matches and len(r.secondary_matches) >= 2)
    v6_matched = sum(1 for r in results if r.secondary_matches and len(r.secondary_matches) >= 3)
    v7_matched = sum(1 for r in results if r.strategy_matches and len(r.strategy_matches) >= 1)

    total_segs = len(results)
    logger.info(f"  Track Coverage:")
    logger.info(f"    V1 (Primary):    {v1_matched:4d}/{total_segs} ({v1_matched/total_segs*100:.1f}%)")
    logger.info(f"    V2 (Alt 1):      {v2_matched:4d}/{total_segs} ({v2_matched/total_segs*100:.1f}%)")
    logger.info(f"    V3 (Alt 2):      {v3_matched:4d}/{total_segs} ({v3_matched/total_segs*100:.1f}%)")
    logger.info(f"    V4 (Sec Pri):    {v4_matched:4d}/{total_segs} ({v4_matched/total_segs*100:.1f}%)")
    logger.info(f"    V5 (Sec Alt 1):  {v5_matched:4d}/{total_segs} ({v5_matched/total_segs*100:.1f}%)")
    logger.info(f"    V6 (Sec Alt 2):  {v6_matched:4d}/{total_segs} ({v6_matched/total_segs*100:.1f}%)")
    logger.info(f"    V7 (Strategy):   {v7_matched:4d}/{total_segs} ({v7_matched/total_segs*100:.1f}%)")

    # Source concentration analysis
    v1_sources: Dict[str, int] = defaultdict(int)
    for r in results:
        if r.primary_match:
            src_name = Path(r.primary_match.video_segment.source_file).stem[:20]
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
    low_alt_segments = sum(1 for r in results if len(r.alternatives or []) < 2)
    no_secondary = sum(1 for r in results if not r.secondary_matches)
    logger.info(f"  Gap Analysis:")
    logger.info(f"    Segments with <2 alternatives: {low_alt_segments}/{total_segs} ({low_alt_segments/total_segs*100:.1f}%)")
    logger.info(f"    Segments with no V4-V6:       {no_secondary}/{total_segs} ({no_secondary/total_segs*100:.1f}%)")

    # Confidence distribution
    confidences = [r.primary_match.confidence for r in results if r.primary_match]
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        high_conf = sum(1 for c in confidences if c >= 0.85)
        med_conf = sum(1 for c in confidences if 0.5 <= c < 0.85)
        low_conf = sum(1 for c in confidences if c < 0.5)
        logger.info(f"  Confidence Distribution (V1):")
        logger.info(f"    Average: {avg_conf:.2f}")
        logger.info(f"    High (≥0.85): {high_conf} | Medium (0.5-0.85): {med_conf} | Low (<0.5): {low_conf}")

    logger.info("=" * 60)

    return results
