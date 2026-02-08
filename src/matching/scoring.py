"""
Confidence scoring and adjustment functions.

Migrated from TieredMatcher methods in matching.py (lines 548-960).
Provides confidence adjustments for:
- Duration mismatch penalties
- Topic-based penalties for chapter matching
- B-roll footage boosts
- Current project vs global cache scoring
- Adaptive thresholds based on voiceover length and candidate variance
"""

from typing import Any, Tuple, List, Optional, TYPE_CHECKING, TYPE_CHECKING as TC

if TC:
    from ..utils import Match
import logging
import math
import statistics

from ..utils import SRTSegment
from ..topic_extraction import compute_topic_penalty

if TYPE_CHECKING:
    from ..config import Config
    from ..config.sections.matching import MatchingScoringConfig

logger = logging.getLogger(__name__)


def _get_scoring_config(config=None) -> Optional['MatchingScoringConfig']:
    """Get the scoring config from a config object, or None if unavailable.

    Returns None if the config doesn't have a proper scoring config.
    Uses duck-typing check to avoid returning Mock auto-generated attributes.
    """
    if config is None:
        return None
    mc = getattr(config, 'matching', None)
    if mc is None:
        return None
    sc = getattr(mc, 'scoring', None)
    if sc is None:
        return None
    # Duck-type check: real MatchingScoringConfig has confidence_floor as a float
    cf = getattr(sc, 'confidence_floor', None)
    if not isinstance(cf, (int, float)):
        return None
    return sc


def calculate_adaptive_threshold(
    base_threshold: float,
    voiceover_text: str,
    candidates: List[Tuple[SRTSegment, float]],
    config=None
) -> Tuple[float, str]:
    """
    Calculate an adaptive threshold based on voiceover length and candidate variance.

    Adjustments:
    - Short voiceover (<20 chars): +0.05 threshold (harder to match, need higher confidence)
    - Low candidate variance (<0.05): -0.05 threshold (clear winner, can accept lower)

    Args:
        base_threshold: The base skip_llm_threshold from config
        voiceover_text: The voiceover segment text
        candidates: List of (video_segment, similarity) tuples
        config: Optional config for additional settings

    Returns:
        Tuple of (adjusted_threshold, adjustment_reason)
    """
    adjustment = 0.0
    reasons = []

    # Adjustment 1: Voiceover length
    # Short voiceover segments are harder to match accurately
    # Require higher confidence to skip LLM for short text
    vo_length = len(voiceover_text.strip()) if voiceover_text else 0
    if vo_length < 20:
        adjustment += 0.05
        reasons.append(f"short_vo({vo_length}c):+0.05")

    # Adjustment 2: Candidate variance
    # Low variance means one candidate is clearly better than others
    # Can accept lower threshold when there's a clear winner
    # US-53-011: Use top-10 when pool >= 10, apply variance floor (0.02)
    if candidates and len(candidates) >= 2:
        # Use top-10 candidates for variance when pool is large enough
        top_n = 10 if len(candidates) >= 10 else 5
        # Convert to Python float to avoid numpy.float32 coercion error in statistics.stdev
        top_scores = [float(sim) for _, sim in candidates[:top_n]]
        try:
            variance = statistics.stdev(top_scores) if len(top_scores) >= 2 else 0.0
        except statistics.StatisticsError:
            variance = 0.0

        # Guard against NaN/Inf from degenerate inputs
        if math.isnan(variance) or math.isinf(variance):
            logger.debug(f"Variance computation returned {variance}, treating as 0.0")
            variance = 0.0

        # Variance floor: treat very low variance as 'uncertain' (false certainty)
        # When top candidates all score nearly the same, it's NOT a clear winner
        VARIANCE_FLOOR = 0.02
        variance_below_floor = variance < VARIANCE_FLOOR

        logger.debug(
            f"Variance computation: candidate_count={len(candidates)}, "
            f"top_n_used={min(top_n, len(candidates))}, "
            f"computed_variance={variance:.4f}, "
            f"below_floor={variance_below_floor}"
        )

        if variance < 0.05 and not variance_below_floor:
            # Low variance but above floor: genuine clear winner
            adjustment -= 0.05
            reasons.append(f"low_var({variance:.3f}):-0.05")
        elif variance_below_floor:
            # Below floor: false certainty, do NOT reduce threshold
            reasons.append(f"var_floor({variance:.3f}):no_adjust")

    # Calculate final threshold, clamped to valid range [0.5, 0.99]
    adjusted_threshold = max(0.5, min(0.99, base_threshold + adjustment))

    reason = "; ".join(reasons) if reasons else "no_adjustment"

    logger.debug(
        f"Adaptive threshold: base={base_threshold:.2f}, "
        f"adjustment={adjustment:+.2f}, final={adjusted_threshold:.2f} ({reason})"
    )

    return adjusted_threshold, reason


def apply_duration_penalty(confidence: float, speed_ratio: float, config) -> float:
    """
    Apply duration-based penalty from config.

    Migrated from TieredMatcher._apply_duration_penalty (lines 548-560).

    Args:
        confidence: Original confidence score
        speed_ratio: Ratio of video duration to voiceover duration
        config: Matching config with ideal_speed_range, soft_speed_range, duration_penalty_factor

    Returns:
        Adjusted confidence score
    """
    mc = config.matching
    ideal_min, ideal_max = mc.ideal_speed_range
    soft_min, soft_max = mc.soft_speed_range

    if ideal_min <= speed_ratio <= ideal_max:
        return confidence  # No penalty
    elif soft_min <= speed_ratio <= soft_max:
        return confidence - mc.duration_penalty_factor
    else:
        return confidence - (mc.duration_penalty_factor * 2)


def apply_topic_penalty(
    confidence: float,
    vo_segment: SRTSegment,
    video_segment: SRTSegment,
    video_topics: dict,
    chapter_matching_enabled: bool,
    topic_mismatch_penalty: float
) -> Tuple[float, str]:
    """
    Apply topic-based confidence penalty for chapter matching.

    Uses caching to avoid recomputing penalties for the same topic pairs
    across multiple matching iterations.

    Migrated from TieredMatcher._apply_topic_penalty (lines 561-610).

    Args:
        confidence: Original confidence score
        vo_segment: Voiceover segment (may have topics from chapter assignment)
        video_segment: Video segment (used to look up video topics)
        video_topics: Dict mapping video paths to VideoTopics objects
        chapter_matching_enabled: Whether chapter matching is enabled
        topic_mismatch_penalty: Maximum penalty for topic mismatch

    Returns:
        Tuple of (adjusted_confidence, penalty_reason)
    """
    if not chapter_matching_enabled:
        return confidence, ""

    # Get voiceover chapter topics
    vo_topics = getattr(vo_segment, 'topics', [])
    if not vo_topics:
        return confidence, ""

    # Get video topics from the video_topics dict
    video_path = video_segment.source_file
    video_topic_info = video_topics.get(video_path)
    if not video_topic_info:
        return confidence, ""

    video_topics_list = video_topic_info.topics if video_topic_info else []
    if not video_topics_list:
        return confidence, ""

    # Check cache for this topic pair
    try:
        from .similarity_cache import get_topic_penalty_cache, topics_hash
        cache = get_topic_penalty_cache()
        vo_hash = topics_hash(vo_topics)
        vid_hash = topics_hash(video_topics_list)

        cached_penalty = cache.get(vo_hash, vid_hash)
        if cached_penalty is not None:
            if cached_penalty > 0:
                adjusted_confidence = max(0.0, confidence - cached_penalty)
                reason = f"topic mismatch penalty (cached): -{cached_penalty:.2f}"
                return adjusted_confidence, reason
            return confidence, ""
    except ImportError:
        # Cache not available, continue without caching
        cached_penalty = None
        vo_hash = None
        vid_hash = None
        cache = None

    # Compute penalty based on topic mismatch
    penalty = compute_topic_penalty(
        vo_topics=vo_topics,
        video_topics=video_topics_list,
        max_penalty=topic_mismatch_penalty,
        min_overlap=1
    )

    # Cache the result
    if cache is not None and vo_hash is not None and vid_hash is not None:
        cache.put(vo_hash, vid_hash, penalty)

    if penalty > 0:
        adjusted_confidence = max(0.0, confidence - penalty)
        reason = f"topic mismatch penalty: -{penalty:.2f}"
        return adjusted_confidence, reason

    return confidence, ""


def apply_broll_boost(
    confidence: float,
    video_segment: SRTSegment,
    config
) -> Tuple[float, str]:
    """
    Apply confidence boost for B-roll/silent video segments.

    Migrated from TieredMatcher._apply_broll_boost (lines 611-650).

    B-roll footage (videos with no speech) is valuable because:
    - No talking heads to distract from voiceover
    - Pure visual content that matches well with any narration
    - More versatile for different contexts

    Args:
        confidence: Original confidence score
        video_segment: Video segment being considered
        config: Config with matching.broll_boost setting

    Returns:
        Tuple of (boosted_confidence, boost_reason)
    """
    # Check if this is a B-roll segment
    is_broll = getattr(video_segment, 'is_broll', False)

    if not is_broll:
        return confidence, ""

    # Get boost amount from config (default 0.1 = +10% confidence)
    mc = config.matching
    broll_boost = getattr(mc, 'broll_boost', 0.1)

    if broll_boost <= 0:
        return confidence, ""

    boosted = min(1.0, confidence + broll_boost)
    reason = f"B-roll boost: +{broll_boost:.2f}"

    logger.debug(f"B-roll boost applied: {confidence:.2f} -> {boosted:.2f}")

    return boosted, reason


def apply_caption_quality_adjustment(
    confidence: float,
    video_segment: SRTSegment,
    config
) -> Tuple[float, str]:
    """
    Apply confidence adjustment based on caption quality (US-007, US-006).

    Supports two modes (mutually exclusive - multiplicative takes precedence):
    1. Multiplicative weights (US-006): When caption_quality_weights dict is set,
       applies: adjusted = raw_confidence * weight
       Example: {high: 1.0, medium: 0.9, low: 0.75}

    2. Additive boost/penalty (US-007): When caption_quality_weights is None,
       uses high_boost and low_penalty for additive adjustments.

    Only ONE mode applies per match - multiplicative mode takes precedence when
    caption_quality_weights is configured. This prevents cascading penalties from
    both modes running on the same match.

    Caption quality levels:
    - 'high': Human-uploaded captions
    - 'medium': Auto-generated captions
    - 'low': Missing/fallback/sparse captions

    Args:
        confidence: Original confidence score
        video_segment: Video segment being considered (may have caption_quality)
        config: Config with caption_quality adjustment settings

    Returns:
        Tuple of (adjusted_confidence, adjustment_reason)
    """
    mc = config.matching

    # Check if caption quality adjustment is enabled
    if not getattr(mc, 'caption_quality_adjustment_enabled', True):
        return confidence, ""

    # Get caption quality from video segment metadata
    caption_quality = getattr(video_segment, 'caption_quality', None)

    if not caption_quality:
        return confidence, ""

    # US-006: Check for multiplicative weights mode
    quality_weights = getattr(mc, 'caption_quality_weights', None)

    if quality_weights is not None and isinstance(quality_weights, dict):
        # Multiplicative weights mode (US-006) - takes precedence over additive
        # Default weights if not specified: high=1.0, medium=0.9, low=0.75
        default_weights = {'high': 1.0, 'medium': 0.9, 'low': 0.75}
        weight = quality_weights.get(caption_quality, default_weights.get(caption_quality, 1.0))

        if weight != 1.0:
            adjusted = max(0.0, min(1.0, confidence * weight))
            reason = f"caption quality {caption_quality}: x{weight:.2f} (multiplicative)"
            # US-006: Specific log format requested
            logger.info(f"Confidence adjusted {confidence:.2f} -> {adjusted:.2f} ({caption_quality} quality caption)")
            return adjusted, reason

        return confidence, ""

    # Legacy additive mode (US-007) - only when caption_quality_weights is None
    # This ensures mutual exclusivity: only one mode applies per match
    high_boost = getattr(mc, 'caption_quality_high_boost', 0.05)
    low_penalty = getattr(mc, 'caption_quality_low_penalty', 0.1)

    if caption_quality == 'high' and high_boost > 0:
        adjusted = min(1.0, confidence + high_boost)
        reason = f"caption quality high: +{high_boost:.2f} (additive)"
        logger.debug(f"Caption quality boost applied: {confidence:.2f} -> {adjusted:.2f}")
        return adjusted, reason

    elif caption_quality == 'low' and low_penalty > 0:
        adjusted = max(0.0, confidence - low_penalty)
        reason = f"caption quality low: -{low_penalty:.2f} (additive)"
        logger.debug(f"Caption quality penalty applied: {confidence:.2f} -> {adjusted:.2f}")
        return adjusted, reason

    # Medium quality or unknown - no adjustment
    return confidence, ""


def apply_tiered_caption_penalties(
    confidence: float,
    video_segment: SRTSegment,
    config
) -> Tuple[float, List[dict]]:
    """
    Apply graduated caption quality penalties based on specific quality issues (US-73-006).

    Checks for individual quality issues on the segment (auto_generated, low_quality,
    missing_timing) and applies separate penalties that stack up to a configurable cap.

    Each penalty appears as a separate entry in the confidence breakdown for transparency.

    Args:
        confidence: Current confidence score (after other adjustments)
        video_segment: Video segment with potential caption_quality_issues metadata
        config: Config with tiered penalty settings

    Returns:
        Tuple of (adjusted_confidence, list of breakdown entries)
    """
    mc = config.matching

    if not getattr(mc, 'caption_quality_adjustment_enabled', True):
        return confidence, []

    # Get caption quality issues list from segment metadata
    caption_quality_issues = getattr(video_segment, 'caption_quality_issues', None)
    if not caption_quality_issues:
        return confidence, []

    # Penalty values from config
    penalty_map = {
        'auto_generated': (
            getattr(mc, 'caption_penalty_auto_generated', -0.05),
            'caption_quality_auto',
        ),
        'low_quality': (
            getattr(mc, 'caption_penalty_low_quality', -0.08),
            'caption_quality_low',
        ),
        'missing_timing': (
            getattr(mc, 'caption_penalty_missing_timing', -0.03),
            'caption_timing_gap',
        ),
    }

    max_penalty = getattr(mc, 'max_caption_penalty', -0.12)

    total_penalty = 0.0
    breakdown_entries = []

    for issue in caption_quality_issues:
        if issue in penalty_map:
            penalty_value, component_name = penalty_map[issue]
            total_penalty += penalty_value

    # Clamp total penalty to max (both are negative, so use max() to limit)
    if total_penalty < max_penalty:
        total_penalty = max_penalty

    if total_penalty == 0.0:
        return confidence, []

    # Build breakdown entries for each contributing penalty (proportionally scaled if capped)
    raw_total = sum(
        penalty_map[issue][0] for issue in caption_quality_issues if issue in penalty_map
    )
    scale = total_penalty / raw_total if raw_total != 0.0 else 1.0

    for issue in caption_quality_issues:
        if issue in penalty_map:
            penalty_value, component_name = penalty_map[issue]
            scaled_penalty = penalty_value * scale
            breakdown_entries.append({
                'component': component_name,
                'adjustment': round(scaled_penalty, 4),
                'reason': f"caption {issue}: {scaled_penalty:+.4f} (tiered)"
            })

    adjusted = max(0.0, min(1.0, confidence + total_penalty))
    logger.debug(
        f"Tiered caption penalties applied: {confidence:.2f} -> {adjusted:.2f} "
        f"(total: {total_penalty:+.2f}, issues: {caption_quality_issues})"
    )

    return adjusted, breakdown_entries


def apply_timing_penalty(
    confidence: float,
    video_segment: SRTSegment,
    config
) -> Tuple[float, str]:
    """
    Apply confidence penalty based on caption timing validation (US-008 Sprint 7).

    When caption timing is poor (low coverage or exceeds video duration),
    apply a multiplicative penalty to reduce match confidence. The penalty
    was calculated at caption fetch time using:

    penalty = 1.0 - (exceeds_ratio * 0.3) - ((1 - coverage_ratio) * 0.2)

    Examples:
        - Perfect timing (100% coverage, no exceeds): penalty = 1.0 (no change)
        - 50% coverage, 20% exceeds: penalty = 0.84 (~16% reduction)
        - 80% coverage, no exceeds: penalty = 0.96 (~4% reduction)

    Args:
        confidence: Current confidence score (may already be adjusted by caption quality)
        video_segment: Video segment with timing_penalty attribute
        config: Config with matching.apply_timing_penalty setting

    Returns:
        Tuple of (adjusted_confidence, adjustment_reason)
    """
    mc = config.matching

    # Check if timing penalty is enabled
    if not getattr(mc, 'apply_timing_penalty', True):
        return confidence, ""

    # Get timing penalty from video segment (set by CaptionStage)
    timing_penalty = getattr(video_segment, 'timing_penalty', None)

    # No penalty attribute or perfect timing - no adjustment
    if timing_penalty is None or timing_penalty >= 1.0:
        return confidence, ""

    # Apply multiplicative penalty
    adjusted = max(0.0, min(1.0, confidence * timing_penalty))
    penalty_pct = (1.0 - timing_penalty) * 100

    reason = f"timing penalty: x{timing_penalty:.2f} (-{penalty_pct:.0f}%)"
    logger.info(f"Timing penalty applied: {confidence:.2f} -> {adjusted:.2f} ({reason})")

    return adjusted, reason


def apply_current_project_boost(
    confidence: float,
    video_segment: SRTSegment,
    config
) -> Tuple[float, str]:
    """
    Apply confidence boost for videos from the current project.

    Migrated from TieredMatcher._apply_current_project_boost (lines 651-690).

    Videos downloaded for the current project are prioritized over
    videos from the global cache (past projects).

    Args:
        confidence: Original confidence score
        video_segment: Video segment being considered
        config: Config with global_cache.current_project_boost setting

    Returns:
        Tuple of (boosted_confidence, boost_reason)
    """
    # Check if this is from global cache
    source = getattr(video_segment, 'source', None)

    # If no source attribute, assume it's from current project
    if source is None or source != 'global_cache':
        return confidence, ""

    # Videos from global cache get a penalty (current project gets relative boost)
    gc_config = getattr(config, 'global_cache', None)
    current_project_boost = getattr(gc_config, 'current_project_boost', 0.1) if gc_config else 0.1

    if current_project_boost <= 0:
        return confidence, ""

    # Apply penalty to global cache videos (equivalent to boosting current project)
    penalized = max(0.0, confidence - current_project_boost)
    reason = f"global cache: -{current_project_boost:.2f}"

    logger.debug(f"Global cache penalty applied: {confidence:.2f} -> {penalized:.2f}")

    return penalized, reason


def compute_duration_penalty(vo_segment: SRTSegment, video_segment: SRTSegment, config) -> float:
    """
    Compute confidence penalty based on speed change required.

    Migrated from TieredMatcher._compute_duration_penalty (lines 902-940).

    Args:
        vo_segment: Voiceover segment (target duration)
        video_segment: Video segment (source duration)
        config: Matching config with duration scoring settings

    Returns:
        Penalty value (0.0 = no penalty, higher = worse)
    """
    mc = config.matching

    if not mc.duration_scoring_enabled:
        return 0.0

    vo_duration = vo_segment.end_time - vo_segment.start_time
    vid_duration = video_segment.end_time - video_segment.start_time

    if vo_duration <= 0 or vid_duration <= 0:
        return 0.0

    # Speed ratio (1.0 = no change, >1 = speed up, <1 = slow down)
    speed_ratio = vid_duration / vo_duration

    ideal_min, ideal_max = mc.ideal_speed_range
    soft_min, soft_max = mc.soft_penalty_range
    penalty_factor = mc.duration_penalty_factor

    if ideal_min <= speed_ratio <= ideal_max:
        # Within ideal range - no penalty
        return 0.0
    elif soft_min <= speed_ratio <= soft_max:
        # Within soft penalty range - small penalty
        return penalty_factor
    else:
        # Outside both ranges - larger penalty
        return penalty_factor * 2


def apply_duration_scoring(
    vo_segment: SRTSegment,
    candidates: List[Tuple[SRTSegment, float]],
    config
) -> List[Tuple[SRTSegment, float, float]]:
    """
    Apply duration scoring to candidates.

    Migrated from TieredMatcher._apply_duration_scoring (lines 941-960).

    Args:
        vo_segment: Voiceover segment
        candidates: List of (video_segment, similarity) tuples
        config: Matching config

    Returns:
        List of (segment, adjusted_similarity, duration_penalty) tuples, sorted by adjusted score
    """
    result = []
    for seg, sim in candidates:
        penalty = compute_duration_penalty(vo_segment, seg, config)
        adjusted = max(0.0, sim - penalty)
        result.append((seg, adjusted, penalty))

    # Re-sort by adjusted similarity
    result.sort(key=lambda x: x[1], reverse=True)
    return result


def compute_temporal_coherence(
    confidence: float,
    video_segment: SRTSegment,
    previous_match: Optional[SRTSegment],
    next_match: Optional[SRTSegment],
    config
) -> Tuple[float, str]:
    """
    Score based on visual/topic similarity to adjacent segment's matches.

    Encourages temporal coherence in the timeline by applying:
    - Small boost (+0.05 default) for clips from same source video as adjacent
    - Small penalty (-0.05 default) for jarring context switches

    A "jarring context switch" is detected when:
    - Previous match has topics/keywords that are completely different
    - Adjacent segments from same topic shouldn't jump to unrelated content

    Args:
        confidence: Original confidence score
        video_segment: Video segment candidate being scored
        previous_match: Previous segment's matched video (may be None for first segment)
        next_match: Next segment's matched video (may be None for last segment or not yet matched)
        config: Config with matching.temporal_coherence_* settings

    Returns:
        Tuple of (adjusted_confidence, reason_string)
    """
    mc = config.matching

    # Check if temporal coherence is enabled
    temporal_coherence_enabled = getattr(mc, 'temporal_coherence_enabled', True)
    if not temporal_coherence_enabled:
        return confidence, ""

    same_source_boost = getattr(mc, 'temporal_coherence_same_source_boost', 0.05)
    context_switch_penalty = getattr(mc, 'temporal_coherence_context_switch_penalty', 0.05)

    # Get current video source file
    current_source = getattr(video_segment, 'source_file', None)
    if not current_source:
        return confidence, ""

    adjustment = 0.0
    reasons = []

    # Check previous match for same-source boost
    if previous_match:
        prev_source = getattr(previous_match, 'source_file', None)
        if prev_source and prev_source == current_source:
            adjustment += same_source_boost
            reasons.append(f"same source as prev: +{same_source_boost:.2f}")
        else:
            # Check for jarring context switch using topics/keywords
            is_jarring = _is_jarring_context_switch(video_segment, previous_match)
            if is_jarring:
                adjustment -= context_switch_penalty
                reasons.append(f"context switch from prev: -{context_switch_penalty:.2f}")

    # Check next match for same-source boost (if available)
    if next_match:
        next_source = getattr(next_match, 'source_file', None)
        if next_source and next_source == current_source:
            adjustment += same_source_boost
            reasons.append(f"same source as next: +{same_source_boost:.2f}")
        else:
            # Check for jarring context switch
            is_jarring = _is_jarring_context_switch(video_segment, next_match)
            if is_jarring:
                adjustment -= context_switch_penalty
                reasons.append(f"context switch to next: -{context_switch_penalty:.2f}")

    if adjustment == 0.0:
        return confidence, ""

    # Clamp to valid range
    adjusted_confidence = max(0.0, min(1.0, confidence + adjustment))
    reason = f"temporal coherence: {'; '.join(reasons)}"

    logger.debug(f"Temporal coherence adjustment: {confidence:.2f} -> {adjusted_confidence:.2f}")

    return adjusted_confidence, reason


def _is_jarring_context_switch(
    current_segment: SRTSegment,
    adjacent_segment: SRTSegment
) -> bool:
    """
    Detect if switching between segments would be jarring (no topic overlap).

    A context switch is considered jarring when:
    - Segments have topics/keywords and they share NO common keywords
    - Both segments must have at least one topic/keyword for this check

    Args:
        current_segment: Current video segment candidate
        adjacent_segment: Adjacent matched segment

    Returns:
        True if context switch is jarring, False otherwise
    """
    # Get topics/keywords from both segments
    current_topics = set(getattr(current_segment, 'topics', []) or [])
    current_keywords = set(getattr(current_segment, 'keywords', []) or [])
    current_all = current_topics | current_keywords

    adjacent_topics = set(getattr(adjacent_segment, 'topics', []) or [])
    adjacent_keywords = set(getattr(adjacent_segment, 'keywords', []) or [])
    adjacent_all = adjacent_topics | adjacent_keywords

    # Both need content for comparison
    if not current_all or not adjacent_all:
        return False

    # Normalize to lowercase for comparison
    current_lower = {k.lower() for k in current_all if k}
    adjacent_lower = {k.lower() for k in adjacent_all if k}

    # Check for any overlap
    overlap = current_lower & adjacent_lower

    # Jarring = no overlap at all
    return len(overlap) == 0


def apply_entity_match_boost(
    confidence: float,
    vo_segment: SRTSegment,
    video_segment: SRTSegment,
    config=None
) -> Tuple[float, str, List[str]]:
    """
    Apply confidence boost when video contains same named entities as voiceover.

    Named entities (people, places, organizations) are strong signals for
    video-voiceover matching. A video mentioning the same person or place
    as the voiceover is highly relevant.

    US-63-011: Entity boost is now configurable via matching.entity_match_boost.

    Graduated boost values (when using legacy config):
    - 1 matching entity: +0.05
    - 2 matching entities: +0.08
    - 3+ matching entities: +0.12

    New mode: If entity_match_boost is set in config, uses that fixed boost value.

    Args:
        confidence: Original confidence score
        vo_segment: Voiceover segment (may have entities from analysis)
        video_segment: Video segment being considered
        config: Config with matching.entity_match_boost setting

    Returns:
        Tuple of (boosted_confidence, boost_reason, matched_entities)
    """
    # Extract entity texts from voiceover segment
    vo_entities = _extract_entity_texts(vo_segment)
    if not vo_entities:
        logger.debug("No entities found in voiceover segment")
        return confidence, "", []

    # Extract entity texts from video segment
    video_entities = _extract_entity_texts(video_segment)
    if not video_entities:
        logger.debug("No entities found in video segment")
        return confidence, "", []

    # Log extracted entities at DEBUG level (US-63-011)
    logger.debug(f"Voiceover entities extracted: {vo_entities}")
    logger.debug(f"Video entities extracted: {video_entities}")

    # Find matching entities (case-insensitive)
    vo_lower = {e.lower() for e in vo_entities}
    video_lower = {e.lower() for e in video_entities}
    matching = vo_lower & video_lower

    if not matching:
        logger.debug(f"No entity matches found between voiceover and video")
        return confidence, "", []

    # Get original-case matched entity names for return
    matched_entities = [e for e in vo_entities if e.lower() in matching]
    match_count = len(matching)

    # Log entity matches at DEBUG level (US-63-011)
    logger.debug(
        f"Entity matches found: {matched_entities} "
        f"(count: {match_count})"
    )

    # US-63-011: Check for configurable entity_match_boost first
    mc = getattr(config, 'matching', None) if config else None
    entity_match_boost = getattr(mc, 'entity_match_boost', None) if mc else None

    if entity_match_boost is not None and entity_match_boost > 0:
        # Use fixed configurable boost (US-63-011)
        boost = entity_match_boost
        logger.debug(
            f"Using configurable entity_match_boost: {boost:.2f} "
            f"for {match_count} matching entities"
        )
    else:
        # Graduated boost based on match count (configurable via scoring config)
        sc = _get_scoring_config(config)
        boosts = getattr(sc, 'entity_match_boosts', None) if sc else None

        if boosts:
            if match_count >= 3:
                boost = boosts.get('3+', 0.12)
            elif match_count == 2:
                boost = boosts.get('2', 0.08)
            else:
                boost = boosts.get('1', 0.05)
        else:
            if match_count >= 3:
                boost = 0.12
            elif match_count == 2:
                boost = 0.08
            else:
                boost = 0.05

    # Apply boost (cap at 1.0)
    boosted = min(1.0, confidence + boost)

    reason = f"entity match: +{boost:.2f} ({match_count} entities: {', '.join(matched_entities[:3])})"

    logger.debug(f"Entity match boost applied: {confidence:.2f} -> {boosted:.2f} ({matched_entities})")

    return boosted, reason, matched_entities


def _extract_entity_texts(segment: SRTSegment) -> List[str]:
    """
    Extract entity text values from a segment.

    Entities are stored as dicts with 'text', 'type', and 'context' keys.
    Also checks keywords list for entity-like entries.
    US-63-011: Also extracts named entities from segment text using heuristic NER.

    Args:
        segment: SRTSegment to extract entities from

    Returns:
        List of entity text values (names)
    """
    entities = []
    seen_lower = set()  # Track seen entities to avoid duplicates

    # Get entities from the entities field
    segment_entities = getattr(segment, 'entities', []) or []
    for entity in segment_entities:
        if isinstance(entity, dict):
            text = entity.get('text', '')
            if text and len(text) >= 2:  # Skip very short entities
                if text.lower() not in seen_lower:
                    entities.append(text)
                    seen_lower.add(text.lower())
        elif isinstance(entity, str):
            if entity and len(entity) >= 2:
                if entity.lower() not in seen_lower:
                    entities.append(entity)
                    seen_lower.add(entity.lower())

    # Also check keywords for entity-like entries (proper nouns, capitalized words)
    keywords = getattr(segment, 'keywords', []) or []
    for kw in keywords:
        if kw and len(kw) >= 2:
            # Check if it looks like a proper noun (capitalized, multi-word)
            if _looks_like_entity(kw) and kw.lower() not in seen_lower:
                entities.append(kw)
                seen_lower.add(kw.lower())

    # US-63-011: Extract named entities from segment text if we have text
    segment_text = getattr(segment, 'text', '') or ''
    if segment_text:
        text_entities = extract_named_entities_from_text(segment_text)
        for entity in text_entities:
            if entity.lower() not in seen_lower:
                entities.append(entity)
                seen_lower.add(entity.lower())

    return entities


def _looks_like_entity(text: str) -> bool:
    """
    Check if a keyword looks like a named entity.

    Named entities typically:
    - Start with capital letter
    - Are proper nouns (person names, place names, organization names)
    - May contain multiple capitalized words

    Args:
        text: Keyword text to check

    Returns:
        True if text looks like a named entity
    """
    if not text:
        return False

    words = text.split()
    if not words:
        return False

    # Check if first word starts with capital
    first_word = words[0]
    if not first_word or not first_word[0].isupper():
        return False

    # Multi-word capitalized phrases are likely entities
    if len(words) > 1:
        # Check if most words are capitalized
        capitalized_count = sum(1 for w in words if w and w[0].isupper())
        return capitalized_count >= len(words) // 2 + 1

    # Single capitalized word - could be entity if not a common word
    # Skip very common words that happen to be capitalized
    common_words = {
        'The', 'A', 'An', 'This', 'That', 'These', 'Those',
        'It', 'They', 'We', 'He', 'She', 'You', 'I',
        'Is', 'Are', 'Was', 'Were', 'Be', 'Been', 'Being',
        'Have', 'Has', 'Had', 'Do', 'Does', 'Did',
        'Will', 'Would', 'Could', 'Should', 'May', 'Might',
        'Can', 'Must', 'Shall'
    }

    return text not in common_words


def extract_named_entities_from_text(text: str) -> List[str]:
    """
    Extract named entities (people, places, organizations) from raw text.

    Uses heuristic-based NER without requiring external libraries.
    Looks for:
    - Capitalized multi-word phrases (e.g., "New York", "Elon Musk")
    - Known geographic patterns (e.g., "City", "State", "Country" suffixes)
    - Organization patterns (e.g., "Inc", "Corp", "LLC", "University")

    Args:
        text: Raw text to extract entities from

    Returns:
        List of entity strings found in the text

    US-63-011: Entity match boost for named entities in voiceover
    """
    if not text:
        return []

    entities = []

    # Common words that shouldn't be entities (expanded list)
    stopwords = {
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        'it', 'they', 'we', 'he', 'she', 'you', 'i', 'me', 'my',
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might',
        'can', 'must', 'shall', 'but', 'and', 'or', 'if', 'so',
        'for', 'with', 'to', 'from', 'of', 'on', 'in', 'at', 'by',
        'about', 'after', 'before', 'during', 'through', 'between',
        'just', 'then', 'now', 'here', 'there', 'where', 'when',
        'what', 'which', 'who', 'how', 'why', 'all', 'each', 'every',
        'some', 'any', 'no', 'not', 'only', 'same', 'other', 'such',
        'more', 'most', 'many', 'much', 'very', 'too', 'also',
        # Sentence starters that aren't entities
        'however', 'therefore', 'furthermore', 'meanwhile',
        'although', 'because', 'since', 'while', 'until',
        'today', 'yesterday', 'tomorrow', 'now', 'then',
    }

    # Geographic suffixes that indicate place names
    geo_suffixes = {
        'city', 'county', 'state', 'province', 'country',
        'island', 'islands', 'mountain', 'mountains', 'river', 'lake',
        'bay', 'valley', 'beach', 'park', 'forest',
        'street', 'avenue', 'boulevard', 'road', 'highway'
    }

    # Organization suffixes
    org_suffixes = {
        'inc', 'inc.', 'corp', 'corp.', 'corporation',
        'llc', 'ltd', 'ltd.', 'limited',
        'co', 'co.', 'company', 'companies',
        'university', 'college', 'institute', 'school',
        'hospital', 'foundation', 'association', 'organization',
        'bank', 'group', 'international'
    }

    # Well-known place name patterns to extract
    city_patterns = {
        'new york', 'los angeles', 'san francisco', 'san diego', 'las vegas',
        'new orleans', 'hong kong', 'buenos aires', 'rio de janeiro',
        'sao paulo', 'el paso', 'santa fe', 'santa barbara', 'st louis',
        'st. louis', 'new delhi', 'tel aviv', 'kuala lumpur'
    }

    # Normalize text for pattern matching
    text_lower = text.lower()

    # Extract known multi-word city names first
    for pattern in city_patterns:
        if pattern in text_lower:
            # Find the original case version
            start_idx = text_lower.find(pattern)
            end_idx = start_idx + len(pattern)
            original = text[start_idx:end_idx]
            entities.append(original.strip())

    # Tokenize preserving case
    words = text.split()

    # Look for capitalized sequences (potential named entities)
    i = 0
    while i < len(words):
        word = words[i].strip('.,!?:;"\'()[]{}')

        if not word:
            i += 1
            continue

        # Check if word starts with capital
        if word[0].isupper() and len(word) >= 2:
            # Check if it's a stopword
            if word.lower() in stopwords:
                i += 1
                continue

            # Collect consecutive capitalized words
            entity_words = [word]
            j = i + 1

            while j < len(words):
                next_word = words[j].strip('.,!?:;"\'()[]{}')
                if not next_word:
                    j += 1
                    continue

                # Check if next word is capitalized or is a connector
                if next_word[0].isupper() and len(next_word) >= 2:
                    entity_words.append(next_word)
                    j += 1
                elif next_word.lower() in {'of', 'the', 'and', 'de', 'la', 'el'}:
                    # Connectors in multi-word entities
                    if j + 1 < len(words):
                        following = words[j + 1].strip('.,!?:;"\'()[]{}')
                        if following and following[0].isupper():
                            entity_words.append(next_word)
                            j += 1
                            continue
                    break
                else:
                    break

            # Create entity from collected words
            if entity_words:
                entity = ' '.join(entity_words)

                # Check if it's a valid entity (not just a sentence starter)
                is_valid = False
                entity_lower = entity.lower()

                # Multi-word capitalized phrases are likely entities
                if len(entity_words) > 1:
                    is_valid = True
                # Single words with geo/org suffixes
                elif any(entity_lower.endswith(suffix) for suffix in geo_suffixes | org_suffixes):
                    is_valid = True
                # Single capitalized word not a stopword and not at sentence start
                elif i > 0:
                    # Not at sentence start
                    prev_word = words[i - 1].strip()
                    if prev_word and not prev_word.endswith(('.', '!', '?', ':')):
                        is_valid = True

                if is_valid and entity not in entities:
                    entities.append(entity)
                    logger.debug(f"Extracted entity from text: '{entity}'")

            i = j
        else:
            i += 1

    return entities


# Transcript quality scoring thresholds
TRANSCRIPT_QUALITY_HIGH_THRESHOLD = 0.8
TRANSCRIPT_QUALITY_MEDIUM_THRESHOLD = 0.5
TRANSCRIPT_MIN_WORD_COUNT_GOOD = 50
TRANSCRIPT_MIN_WORD_COUNT_MEDIUM = 20


def calculate_transcript_quality(
    transcript_text: str,
    config=None
) -> Tuple[float, str, str]:
    """
    Calculate the quality score for a video transcript.

    Quality is assessed based on:
    - Word count (>50 = good, 20-50 = medium, <20 = low)
    - Sentence coherence (proper sentence structure with punctuation)
    - Language consistency (mixed languages or gibberish detection)

    Args:
        transcript_text: The transcript text to evaluate
        config: Optional config for additional settings

    Returns:
        Tuple of (quality_score, quality_tier, reason)
        - quality_score: Float between 0.0 and 1.0
        - quality_tier: "high" (>0.8), "medium" (0.5-0.8), or "low" (<0.5)
        - reason: String explaining the quality assessment
    """
    if not transcript_text or not transcript_text.strip():
        return 0.0, "low", "empty_transcript"

    text = transcript_text.strip()
    reasons = []

    # Read configurable thresholds (fall back to module constants)
    sc = _get_scoring_config(config)
    min_words_good = getattr(sc, 'transcript_min_words_good', TRANSCRIPT_MIN_WORD_COUNT_GOOD) if sc else TRANSCRIPT_MIN_WORD_COUNT_GOOD
    min_words_medium = getattr(sc, 'transcript_min_words_medium', TRANSCRIPT_MIN_WORD_COUNT_MEDIUM) if sc else TRANSCRIPT_MIN_WORD_COUNT_MEDIUM

    # Factor 1: Word count scoring (0.0 - 0.4)
    # More words generally means better quality transcription
    words = text.split()
    word_count = len(words)

    if word_count >= min_words_good:
        word_score = 0.4
    elif word_count >= min_words_medium:
        # Linear interpolation between medium and good word counts
        word_score = 0.2 + 0.2 * ((word_count - min_words_medium) /
                                   (min_words_good - min_words_medium))
    else:
        # Linear interpolation between 0 and medium word count
        word_score = 0.2 * (word_count / min_words_medium) if word_count > 0 else 0.0

    reasons.append(f"words:{word_count}")

    # Factor 2: Sentence coherence scoring (0.0 - 0.35)
    # Check for proper sentence structure with punctuation
    coherence_score = _calculate_sentence_coherence(text, words)
    if coherence_score < 0.2:
        reasons.append("low_coherence")
    elif coherence_score >= 0.3:
        reasons.append("good_coherence")

    # Factor 3: Language consistency scoring (0.0 - 0.25)
    # Check for mixed languages, gibberish, or ASR errors
    consistency_score = _calculate_language_consistency(text, words)
    if consistency_score < 0.15:
        reasons.append("inconsistent_language")
    elif consistency_score >= 0.22:
        reasons.append("consistent_language")

    # Calculate total quality score
    quality_score = word_score + coherence_score + consistency_score

    # Clamp to valid range
    quality_score = max(0.0, min(1.0, quality_score))

    # Determine quality tier (configurable thresholds)
    high_threshold = getattr(sc, 'transcript_quality_high', TRANSCRIPT_QUALITY_HIGH_THRESHOLD) if sc else TRANSCRIPT_QUALITY_HIGH_THRESHOLD
    medium_threshold = getattr(sc, 'transcript_quality_medium', TRANSCRIPT_QUALITY_MEDIUM_THRESHOLD) if sc else TRANSCRIPT_QUALITY_MEDIUM_THRESHOLD
    if quality_score >= high_threshold:
        quality_tier = "high"
    elif quality_score >= medium_threshold:
        quality_tier = "medium"
    else:
        quality_tier = "low"

    reason = "; ".join(reasons)

    logger.debug(
        f"Transcript quality: score={quality_score:.2f}, tier={quality_tier}, "
        f"word={word_score:.2f}, coherence={coherence_score:.2f}, "
        f"consistency={consistency_score:.2f} ({reason})"
    )

    return quality_score, quality_tier, reason


def _calculate_sentence_coherence(text: str, words: List[str]) -> float:
    """
    Calculate sentence coherence score based on punctuation and structure.

    Good transcripts have:
    - Proper sentence-ending punctuation (. ! ?)
    - Reasonable sentence lengths
    - Capitalized sentence beginnings

    Args:
        text: The full transcript text
        words: Pre-split list of words

    Returns:
        Coherence score between 0.0 and 0.35
    """
    if not text or len(words) < 3:
        return 0.0

    score = 0.0

    # Check for sentence-ending punctuation
    sentence_endings = text.count('.') + text.count('!') + text.count('?')
    if sentence_endings > 0:
        # Expect roughly 1 sentence per 10-15 words
        expected_sentences = max(1, len(words) // 12)
        punctuation_ratio = min(1.0, sentence_endings / expected_sentences)
        score += 0.15 * punctuation_ratio

    # Check for capitalized words (sentence beginnings)
    capitalized = sum(1 for w in words if w and w[0].isupper())
    if capitalized > 0:
        # At least some capitalization indicates structure
        cap_ratio = min(1.0, capitalized / max(1, sentence_endings + 1))
        score += 0.10 * min(1.0, cap_ratio)

    # Check for reasonable word lengths (not all short gibberish)
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len >= 4.0:
        score += 0.10
    elif avg_word_len >= 3.0:
        score += 0.05

    return min(0.35, score)


def _calculate_language_consistency(text: str, words: List[str]) -> float:
    """
    Calculate language consistency score to detect mixed languages or ASR errors.

    Detects:
    - Excessive repetition (ASR stuttering)
    - Too many short words (gibberish)
    - Mixed script detection (latin + other)
    - Filler word overload

    Args:
        text: The full transcript text
        words: Pre-split list of words

    Returns:
        Consistency score between 0.0 and 0.25
    """
    if not words or len(words) < 2:
        return 0.0

    score = 0.25  # Start with full score, deduct for issues

    # Check for excessive word repetition (ASR stuttering)
    if len(words) >= 5:
        unique_words = set(w.lower() for w in words)
        repetition_ratio = len(unique_words) / len(words)
        if repetition_ratio < 0.3:
            score -= 0.15  # Heavy repetition
        elif repetition_ratio < 0.5:
            score -= 0.08  # Moderate repetition

    # Check for too many very short words (potential gibberish)
    short_words = sum(1 for w in words if len(w) <= 2)
    short_ratio = short_words / len(words)
    if short_ratio > 0.5:
        score -= 0.10
    elif short_ratio > 0.35:
        score -= 0.05

    # Check for filler word overload
    filler_words = {'um', 'uh', 'er', 'ah', 'like', 'you know', 'basically'}
    filler_count = sum(1 for w in words if w.lower() in filler_words)
    if len(words) > 5:
        filler_ratio = filler_count / len(words)
        if filler_ratio > 0.2:
            score -= 0.08
        elif filler_ratio > 0.1:
            score -= 0.04

    # Check for non-ASCII characters indicating mixed scripts
    # (not necessarily bad, but can indicate ASR confusion)
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if len(text) > 0:
        ascii_ratio = ascii_chars / len(text)
        if ascii_ratio < 0.7:
            score -= 0.05  # Significant non-ASCII content

    return max(0.0, score)


# Multi-modal similarity weighting defaults
# These are normalized weights that must sum to 1.0
DEFAULT_MULTIMODAL_WEIGHTS = {
    'text_embedding': 0.40,     # Embedding similarity weight (40%)
    'keyword_overlap': 0.25,    # Keyword overlap weight (25%)
    'entity_match': 0.20,       # Entity match weight (20%)
    'visual_description': 0.15  # Visual description weight (15%)
}

# Valid weight keys for multimodal scoring
VALID_MULTIMODAL_WEIGHT_KEYS = set(DEFAULT_MULTIMODAL_WEIGHTS.keys())

# Tolerance for weight sum validation
WEIGHT_SUM_TOLERANCE = 0.01


class MultimodalScoringTracker:
    """Tracks multimodal vs embedding-only fallback counts during a matching run.

    US-53-009: Provides summary statistics for scoring component usage.
    """

    def __init__(self):
        self.multimodal_active_count = 0
        self.embedding_only_count = 0

    def record_multimodal(self):
        self.multimodal_active_count += 1

    def record_embedding_only(self):
        self.embedding_only_count += 1

    def reset(self):
        self.multimodal_active_count = 0
        self.embedding_only_count = 0

    def log_summary(self):
        """Log summary of multimodal vs embedding-only match counts."""
        total = self.multimodal_active_count + self.embedding_only_count
        if total == 0:
            return
        logger.info(
            f"Multimodal scoring summary: {self.multimodal_active_count}/{total} matches used "
            f"full multimodal scoring, {self.embedding_only_count}/{total} fell back to "
            f"embedding-only (all non-embedding components were 0)"
        )


# Module-level tracker instance, reset per matching run
_multimodal_tracker = MultimodalScoringTracker()


def get_multimodal_tracker() -> MultimodalScoringTracker:
    """Get the module-level multimodal scoring tracker."""
    return _multimodal_tracker


def validate_multimodal_weights(weights: dict) -> dict:
    """
    Validate and normalize multimodal scoring weights.

    Checks:
    - All weight values are in [0.0, 1.0] range (clamps and warns if not)
    - Weights sum to approximately 1.0 (normalizes and warns if not)

    Args:
        weights: Dict of weight name -> weight value

    Returns:
        Validated and potentially normalized copy of weights
    """
    if not weights:
        return None

    validated = dict(weights)

    # Check for out-of-range values and clamp
    any_clamped = False
    for key, value in validated.items():
        if not isinstance(value, (int, float)):
            logger.warning(
                f"Multimodal weight '{key}' has non-numeric value {value!r}, setting to 0.0"
            )
            validated[key] = 0.0
            any_clamped = True
            continue

        if value < 0.0:
            logger.warning(
                f"Multimodal weight '{key}' is negative ({value:.3f}), clamping to 0.0"
            )
            validated[key] = 0.0
            any_clamped = True
        elif value > 1.0:
            logger.warning(
                f"Multimodal weight '{key}' exceeds 1.0 ({value:.3f}), clamping to 1.0"
            )
            validated[key] = 1.0
            any_clamped = True

    # Check sum and normalize if needed
    weight_sum = sum(validated.values())
    if weight_sum == 0.0:
        logger.warning(
            "All multimodal weights are zero after validation, falling back to defaults"
        )
        return dict(DEFAULT_MULTIMODAL_WEIGHTS)

    if abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
        logger.warning(
            f"Multimodal weights sum to {weight_sum:.3f} (expected ~1.0), "
            f"normalizing: {validated}"
        )
        validated = {k: v / weight_sum for k, v in validated.items()}

    return validated


def compute_multimodal_score(
    embedding_similarity: float,
    keyword_overlap_score: float,
    entity_match_score: float,
    visual_description_score: float,
    weights: dict = None,
    multimodal_enabled: bool = True,
    scoring_config=None
) -> Tuple[float, str, dict]:
    """
    Compute weighted multi-modal similarity score.

    Combines multiple similarity signals into a single score using weighted fusion:
    - text_embedding: Raw embedding similarity (semantic match)
    - keyword_overlap: Normalized keyword overlap between voiceover and video
    - entity_match: Normalized named entity overlap (people, places, organizations)
    - visual_description: Similarity based on visual scene descriptions

    Each component should be normalized to [0, 1] before passing to this function.

    Args:
        embedding_similarity: Text embedding similarity score (0-1)
        keyword_overlap_score: Keyword overlap score (0-1)
        entity_match_score: Entity match score (0-1)
        visual_description_score: Visual description similarity score (0-1)
        weights: Optional dict with weight values (defaults to DEFAULT_MULTIMODAL_WEIGHTS)
        multimodal_enabled: Whether to use multimodal weighting (if False, returns embedding_similarity)

    Returns:
        Tuple of:
        - multimodal_score: Weighted combined score (0-1)
        - reason: Explanation string showing component contributions
        - component_scores: Dict with individual component scores and weights
    """
    if not multimodal_enabled:
        return embedding_similarity, "multimodal_disabled", {
            'embedding_similarity': embedding_similarity,
            'keyword_overlap': keyword_overlap_score,
            'entity_match': entity_match_score,
            'visual_description': visual_description_score,
            'weights_used': None
        }

    # Use default weights if not provided, validate and normalize
    # scoring_config.multimodal_default_weights overrides the module-level constant
    default_weights = DEFAULT_MULTIMODAL_WEIGHTS
    if scoring_config is not None:
        cfg_defaults = getattr(scoring_config, 'multimodal_default_weights', None)
        if isinstance(cfg_defaults, dict):
            default_weights = cfg_defaults

    if weights:
        w = validate_multimodal_weights(weights)
        if w is None:
            w = dict(default_weights)
    else:
        w = dict(default_weights)

    # Final safety check: all-zero weights after validation
    weight_sum = sum(w.values())
    if weight_sum == 0.0:
        logger.warning("All multimodal weights are zero, returning score=0.0")
        return 0.0, "all_weights_zero", {
            'embedding_similarity': embedding_similarity,
            'keyword_overlap': keyword_overlap_score,
            'entity_match': entity_match_score,
            'visual_description': visual_description_score,
            'weights_used': w
        }

    # US-53-009: Check for NaN/Inf before clamping and log WARNING
    component_names = {
        'embedding_similarity': embedding_similarity,
        'keyword_overlap': keyword_overlap_score,
        'entity_match': entity_match_score,
        'visual_description': visual_description_score,
    }
    for comp_name, comp_value in component_names.items():
        if math.isnan(comp_value):
            logger.warning(
                f"Multimodal component '{comp_name}' is NaN (raw value: {comp_value!r}), "
                f"clamping to 0.0"
            )
        elif math.isinf(comp_value):
            clamped_to = 1.0 if comp_value > 0 else 0.0
            logger.warning(
                f"Multimodal component '{comp_name}' is Inf (raw value: {comp_value!r}), "
                f"clamping to {clamped_to}"
            )

    # Clamp input scores to [0, 1] range, handling NaN and Inf
    def safe_clamp(value: float) -> float:
        """Clamp value to [0, 1], converting NaN/Inf to valid values."""
        if math.isnan(value):
            return 0.0
        if math.isinf(value):
            return 1.0 if value > 0 else 0.0
        return max(0.0, min(1.0, value))

    emb_clamped = safe_clamp(embedding_similarity)
    kw_clamped = safe_clamp(keyword_overlap_score)
    ent_clamped = safe_clamp(entity_match_score)
    vis_clamped = safe_clamp(visual_description_score)

    # Compute weighted contributions
    emb_contrib = emb_clamped * w.get('text_embedding', 0.4)
    kw_contrib = kw_clamped * w.get('keyword_overlap', 0.25)
    ent_contrib = ent_clamped * w.get('entity_match', 0.2)
    vis_contrib = vis_clamped * w.get('visual_description', 0.15)

    # Sum weighted contributions
    multimodal_score = emb_contrib + kw_contrib + ent_contrib + vis_contrib

    # Clamp final score to [0, 1]
    multimodal_score = max(0.0, min(1.0, multimodal_score))

    # Build component scores dict for detailed logging
    component_scores = {
        'embedding_similarity': emb_clamped,
        'keyword_overlap': kw_clamped,
        'entity_match': ent_clamped,
        'visual_description': vis_clamped,
        'embedding_contribution': emb_contrib,
        'keyword_contribution': kw_contrib,
        'entity_contribution': ent_contrib,
        'visual_contribution': vis_contrib,
        'weights_used': w
    }

    # Build reason string
    reason_parts = []
    if emb_clamped > 0:
        reason_parts.append(f"emb:{emb_clamped:.2f}*{w.get('text_embedding', 0.4):.0%}={emb_contrib:.3f}")
    if kw_clamped > 0:
        reason_parts.append(f"kw:{kw_clamped:.2f}*{w.get('keyword_overlap', 0.25):.0%}={kw_contrib:.3f}")
    if ent_clamped > 0:
        reason_parts.append(f"ent:{ent_clamped:.2f}*{w.get('entity_match', 0.2):.0%}={ent_contrib:.3f}")
    if vis_clamped > 0:
        reason_parts.append(f"vis:{vis_clamped:.2f}*{w.get('visual_description', 0.15):.0%}={vis_contrib:.3f}")

    reason = f"multimodal({' + '.join(reason_parts)})={multimodal_score:.3f}"

    # US-53-009: DEBUG log component breakdown for every scored match
    logger.debug(
        f"Multimodal component breakdown: "
        f"embedding_similarity={emb_clamped:.3f} (weight={w.get('text_embedding', 0.4):.2f}), "
        f"keyword_overlap={kw_clamped:.3f} (weight={w.get('keyword_overlap', 0.25):.2f}), "
        f"entity_match={ent_clamped:.3f} (weight={w.get('entity_match', 0.2):.2f}), "
        f"visual_similarity={vis_clamped:.3f} (weight={w.get('visual_description', 0.15):.2f}) "
        f"-> score={multimodal_score:.3f}"
    )

    # US-53-009: Track multimodal vs embedding-only and log INFO for fallback
    is_embedding_only = (kw_clamped == 0.0 and ent_clamped == 0.0 and vis_clamped == 0.0)
    if is_embedding_only:
        logger.info(
            f"Multimodal scoring fell back to embedding-only: all non-embedding components "
            f"are 0 (keyword_overlap={kw_clamped}, entity_match={ent_clamped}, "
            f"visual_similarity={vis_clamped}), score={multimodal_score:.3f}"
        )
        _multimodal_tracker.record_embedding_only()
    else:
        _multimodal_tracker.record_multimodal()

    return multimodal_score, reason, component_scores


def calculate_keyword_overlap_score(
    vo_keywords: List[str],
    video_keywords: List[str],
    scoring_config=None
) -> Tuple[float, List[str]]:
    """
    Calculate normalized keyword overlap score between voiceover and video.

    Args:
        vo_keywords: Keywords from voiceover segment
        video_keywords: Keywords from video segment
        scoring_config: Optional MatchingScoringConfig for configurable thresholds

    Returns:
        Tuple of (overlap_score, matched_keywords)
        - overlap_score: Normalized score (0-1) based on number of matches
        - matched_keywords: List of matched keyword strings
    """
    if not vo_keywords or not video_keywords:
        return 0.0, []

    # Normalize to lowercase for comparison
    vo_lower = {k.lower().strip() for k in vo_keywords if k}
    video_lower = {k.lower().strip() for k in video_keywords if k}

    # Find intersection
    matched = vo_lower & video_lower

    if not matched:
        return 0.0, []

    # Normalize score: more matches = higher score, with diminishing returns
    # Configurable thresholds via scoring_config.keyword_overlap_thresholds
    match_count = len(matched)
    thresholds = getattr(scoring_config, 'keyword_overlap_thresholds', None) if scoring_config else None
    if thresholds:
        if match_count >= 5:
            score = thresholds.get('5+', 1.0)
        elif match_count >= 4:
            score = thresholds.get('4', 0.9)
        elif match_count >= 3:
            score = thresholds.get('3', 0.75)
        elif match_count >= 2:
            score = thresholds.get('2', 0.55)
        else:
            score = thresholds.get('1', 0.35)
    else:
        if match_count >= 5:
            score = 1.0
        elif match_count >= 4:
            score = 0.9
        elif match_count >= 3:
            score = 0.75
        elif match_count >= 2:
            score = 0.55
        else:
            score = 0.35

    # Get original-case matched keywords
    matched_original = [k for k in vo_keywords if k.lower().strip() in matched]

    return score, matched_original


def calculate_entity_match_score(
    vo_entities: List[str],
    video_entities: List[str]
) -> Tuple[float, List[str]]:
    """
    Calculate normalized entity match score between voiceover and video.

    Named entities (people, places, organizations) are strong signals for matching.
    A video mentioning the same person or place as the voiceover is highly relevant.

    Args:
        vo_entities: Entity texts from voiceover segment
        video_entities: Entity texts from video segment

    Returns:
        Tuple of (entity_score, matched_entities)
        - entity_score: Normalized score (0-1) based on entity matches
        - matched_entities: List of matched entity strings
    """
    if not vo_entities or not video_entities:
        return 0.0, []

    # Normalize for comparison
    vo_lower = {e.lower().strip() for e in vo_entities if e and len(e) >= 2}
    video_lower = {e.lower().strip() for e in video_entities if e and len(e) >= 2}

    # Find matching entities
    matched = vo_lower & video_lower

    if not matched:
        return 0.0, []

    # Normalize score: entities are strong signals
    # 1 match = 0.6, 2 matches = 0.8, 3+ matches = 1.0
    match_count = len(matched)
    if match_count >= 3:
        score = 1.0
    elif match_count >= 2:
        score = 0.8
    else:
        score = 0.6

    # Get original-case matched entities
    matched_original = [e for e in vo_entities if e.lower().strip() in matched]

    return score, matched_original


def calculate_visual_description_score(
    vo_text: str,
    video_description: str,
    visual_keywords: List[str] = None
) -> float:
    """
    Calculate visual description similarity score.

    Compares voiceover text to video visual description/keywords.
    Useful when video has scene descriptions from vision API.

    Args:
        vo_text: Voiceover segment text
        video_description: Video visual description or transcript
        visual_keywords: Optional list of visual keywords from scene analysis

    Returns:
        Visual similarity score (0-1)
    """
    if not vo_text:
        return 0.0

    if not video_description and not visual_keywords:
        return 0.0

    score = 0.0

    # Extract significant words from voiceover (longer words, no stopwords)
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'must', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        'just', 'but', 'and', 'if', 'or', 'because', 'until', 'while', 'this',
        'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'what',
        'which', 'who', 'whom', 'you', 'your', 'we', 'our', 'he', 'she', 'him',
        'her', 'his', 'i', 'me', 'my'
    }

    vo_words = set(
        w.lower().strip('.,!?:;"\'()[]{}')
        for w in vo_text.split()
        if len(w) >= 4 and w.lower() not in stopwords
    )

    if not vo_words:
        return 0.0

    matched_count = 0
    total_checks = 0

    # Check against video description
    if video_description:
        desc_words = set(
            w.lower().strip('.,!?:;"\'()[]{}')
            for w in video_description.split()
            if len(w) >= 4 and w.lower() not in stopwords
        )

        desc_overlap = vo_words & desc_words
        if desc_words:
            matched_count += len(desc_overlap)
            total_checks += min(len(vo_words), len(desc_words))

    # Check against visual keywords
    if visual_keywords:
        vis_kw_lower = set(k.lower().strip() for k in visual_keywords if k)
        for vo_word in vo_words:
            for vis_kw in vis_kw_lower:
                if vo_word in vis_kw or vis_kw in vo_word:
                    matched_count += 1
                    break
        total_checks += len(vo_words)

    if total_checks == 0:
        return 0.0

    # Calculate raw ratio and apply diminishing returns
    raw_ratio = matched_count / total_checks
    # Scale: 10% overlap = 0.3, 30% = 0.6, 50%+ = 0.85-1.0
    if raw_ratio >= 0.5:
        score = 0.85 + (raw_ratio - 0.5) * 0.3
    elif raw_ratio >= 0.3:
        score = 0.6 + (raw_ratio - 0.3) * 1.25
    elif raw_ratio >= 0.1:
        score = 0.3 + (raw_ratio - 0.1) * 1.5
    else:
        score = raw_ratio * 3.0

    return min(1.0, max(0.0, score))


def adjust_embedding_weight_for_transcript_quality(
    base_weight: float,
    quality_tier: str,
    quality_weight_enabled: bool = True,
    low_quality_reduction: float = 0.20
) -> Tuple[float, str]:
    """
    Adjust embedding weight based on transcript quality tier.

    When transcript quality is low, embedding similarity is less reliable
    because the transcript text doesn't accurately represent video content.
    This function reduces embedding weight for low-quality transcripts.

    Args:
        base_weight: The base embedding weight (e.g., 0.4 for 40%)
        quality_tier: Quality tier from calculate_transcript_quality ("high", "medium", "low")
        quality_weight_enabled: Whether to apply the adjustment (config option)
        low_quality_reduction: Fraction to reduce weight by for low quality (default: 0.20 = 20%)

    Returns:
        Tuple of (adjusted_weight, reason)
    """
    if not quality_weight_enabled:
        return base_weight, "quality_adjustment_disabled"

    if quality_tier == "low":
        # Reduce embedding weight by the specified reduction factor (default 20%)
        reduction = base_weight * low_quality_reduction
        adjusted = base_weight - reduction
        reason = f"low_quality_transcript:-{low_quality_reduction:.0%}"
        logger.debug(f"Embedding weight reduced for low quality transcript: {base_weight:.2f} -> {adjusted:.2f}")
        return adjusted, reason

    elif quality_tier == "medium":
        # Small reduction for medium quality (half of low quality reduction)
        reduction = base_weight * (low_quality_reduction / 2)
        adjusted = base_weight - reduction
        reason = f"medium_quality_transcript:-{low_quality_reduction/2:.0%}"
        return adjusted, reason

    else:  # high quality
        return base_weight, "high_quality_transcript"


# Semantic coherence constants (topic flow between adjacent segments)
SEMANTIC_COHERENCE_SMOOTH_THRESHOLD = 0.6  # Similarity above this = smooth flow
SEMANTIC_COHERENCE_ABRUPT_THRESHOLD = 0.3  # Similarity below this = abrupt transition
SEMANTIC_COHERENCE_SMOOTH_BOOST = 0.03  # Boost for smooth topic flow
SEMANTIC_COHERENCE_ABRUPT_PENALTY = 0.05  # Penalty for abrupt topic flow


def compute_semantic_coherence(
    current_embedding: Any,
    previous_embedding: Any,
    semantic_coherence_enabled: bool = True,
    scoring_config=None
) -> Tuple[float, str]:
    """
    Compute semantic coherence adjustment based on topic flow between adjacent matches.

    Semantic coherence measures how smoothly topics transition between segments.
    A high embedding similarity between current and previous matches indicates
    smooth topic flow (related content), while low similarity indicates an
    abrupt topic change.

    Adjustments:
    - Smooth flow (similarity > 0.6): +0.03 boost (good continuity)
    - Abrupt flow (similarity < 0.3): -0.05 penalty (jarring transition)
    - Neutral (0.3 - 0.6): no adjustment

    Args:
        current_embedding: Embedding vector of current match candidate (numpy array or list)
        previous_embedding: Embedding vector of previous matched segment (numpy array or list)
        semantic_coherence_enabled: Whether to apply semantic coherence adjustment (config option)

    Returns:
        Tuple of (adjustment, reason):
        - adjustment: Float adjustment to apply to confidence (+0.03, 0.0, or -0.05)
        - reason: String explaining the adjustment
    """
    if not semantic_coherence_enabled:
        return 0.0, "semantic_coherence_disabled"

    # Handle None embeddings
    if current_embedding is None or previous_embedding is None:
        return 0.0, "missing_embedding"

    # Import cosine_similarity from embeddings module
    try:
        from ..embeddings import cosine_similarity
    except ImportError:
        logger.warning("Could not import cosine_similarity from embeddings module")
        return 0.0, "cosine_similarity_unavailable"

    # Compute embedding similarity between current and previous
    try:
        similarity = cosine_similarity(current_embedding, previous_embedding)
    except Exception as e:
        logger.warning(f"Failed to compute cosine similarity: {e}")
        return 0.0, f"similarity_error:{str(e)}"

    # Read configurable thresholds (fall back to module constants)
    smooth_threshold = getattr(scoring_config, 'semantic_coherence_smooth_threshold', SEMANTIC_COHERENCE_SMOOTH_THRESHOLD) if scoring_config else SEMANTIC_COHERENCE_SMOOTH_THRESHOLD
    abrupt_threshold = getattr(scoring_config, 'semantic_coherence_abrupt_threshold', SEMANTIC_COHERENCE_ABRUPT_THRESHOLD) if scoring_config else SEMANTIC_COHERENCE_ABRUPT_THRESHOLD
    smooth_boost = getattr(scoring_config, 'semantic_coherence_smooth_boost', SEMANTIC_COHERENCE_SMOOTH_BOOST) if scoring_config else SEMANTIC_COHERENCE_SMOOTH_BOOST
    abrupt_penalty = getattr(scoring_config, 'semantic_coherence_abrupt_penalty', SEMANTIC_COHERENCE_ABRUPT_PENALTY) if scoring_config else SEMANTIC_COHERENCE_ABRUPT_PENALTY

    # Apply adjustments based on similarity thresholds
    if similarity > smooth_threshold:
        adjustment = smooth_boost
        reason = f"smooth_topic_flow(sim={similarity:.3f}):+{smooth_boost}"
    elif similarity < abrupt_threshold:
        adjustment = -abrupt_penalty
        reason = f"abrupt_topic_flow(sim={similarity:.3f}):-{abrupt_penalty}"
    else:
        adjustment = 0.0
        reason = f"neutral_topic_flow(sim={similarity:.3f})"

    logger.debug(
        f"Semantic coherence: similarity={similarity:.3f}, adjustment={adjustment:+.3f} ({reason})"
    )

    return adjustment, reason


def compute_relevance_matrix(
    voiceover_chapter_keywords: List[List[str]],
    video_chapter_keywords: List[List[str]],
) -> List[List[float]]:
    """
    Compute a cross-chapter relevance matrix based on topic keyword overlap (US-71-005).

    Each cell [i][j] is the Jaccard similarity (intersection over union) between
    voiceover chapter i's keywords and video chapter j's keywords, normalized to 0-1.

    Args:
        voiceover_chapter_keywords: List of keyword lists, one per voiceover chapter
        video_chapter_keywords: List of keyword lists, one per video chapter

    Returns:
        2D list of floats (vo_chapters x video_chapters), each in [0.0, 1.0]
    """
    if not voiceover_chapter_keywords or not video_chapter_keywords:
        return []

    matrix = []
    for vo_kw in voiceover_chapter_keywords:
        vo_set = {k.lower() for k in vo_kw} if vo_kw else set()
        row = []
        for vid_kw in video_chapter_keywords:
            vid_set = {k.lower() for k in vid_kw} if vid_kw else set()
            union = vo_set | vid_set
            if not union:
                row.append(0.0)
            else:
                row.append(len(vo_set & vid_set) / len(union))
        matrix.append(row)

    return matrix


# Pool normalization constants
POOL_NORMALIZATION_REFERENCE_SIZE = 50  # Reference pool size for normalization
POOL_NORMALIZATION_MIN_FACTOR = 0.8  # Minimum normalization factor (caps boost)
POOL_NORMALIZATION_MAX_FACTOR = 1.2  # Maximum normalization factor (caps reduction)
POOL_SMALL_THRESHOLD = 10  # Pool considered "small" below this
POOL_LARGE_THRESHOLD = 100  # Pool considered "large" above this
POOL_TIGHT_MARGIN_THRESHOLD = 0.05  # Top-2 score difference threshold for "tight margin"


class MatchScoring:
    """
    Centralized scoring class for match confidence calculations.

    Provides a unified interface for all scoring operations, enabling:
    - Configuration-driven scoring adjustments
    - Easy testing and mocking
    - Composition pattern for TieredMatcher

    Usage:
        scoring = MatchScoring(config)
        confidence, reason, breakdown = scoring.apply_all_adjustments(
            base_confidence, vo_segment, video_segment, video_topics
        )
    """

    def __init__(self, config=None):
        """
        Initialize MatchScoring with configuration.

        Args:
            config: Configuration object with matching settings
        """
        self.config = config
        self._mc = config.matching if config else None
        # Get scoring config with duck-type validation
        _sc_candidate = getattr(self._mc, 'scoring', None) if self._mc else None
        if _sc_candidate is not None and isinstance(getattr(_sc_candidate, 'confidence_floor', None), (int, float)):
            self._sc = _sc_candidate
        else:
            self._sc = None

    def calculate_confidence(
        self,
        embedding_similarity: float,
        keyword_score: float = 0.0,
        entity_score: float = 0.0,
        visual_score: float = 0.0
    ) -> Tuple[float, str, dict]:
        """
        Calculate multimodal confidence from component scores.

        Args:
            embedding_similarity: Raw embedding similarity (0-1)
            keyword_score: Keyword overlap score (0-1)
            entity_score: Entity match score (0-1)
            visual_score: Visual description score (0-1)

        Returns:
            Tuple of (confidence, reason, component_scores)
        """
        weights = getattr(self._mc, 'multimodal_weights', None) if self._mc else None
        multimodal_enabled = getattr(self._mc, 'multimodal_enabled', True) if self._mc else True

        return compute_multimodal_score(
            embedding_similarity=embedding_similarity,
            keyword_overlap_score=keyword_score,
            entity_match_score=entity_score,
            visual_description_score=visual_score,
            weights=weights,
            multimodal_enabled=multimodal_enabled,
            scoring_config=self._sc
        )

    def normalize_score(
        self,
        confidence: float,
        pool_size: int,
        candidates: Optional[List[Tuple[SRTSegment, float]]] = None
    ) -> Tuple[float, str]:
        """
        Normalize confidence score based on candidate pool size.

        Args:
            confidence: Raw confidence score
            pool_size: Number of candidates in pool
            candidates: Optional sorted candidate list

        Returns:
            Tuple of (normalized_confidence, reason)
        """
        pool_norm_enabled = getattr(self._mc, 'pool_normalization_enabled', True) if self._mc else True

        return normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=pool_size,
            candidates=candidates,
            pool_normalization_enabled=pool_norm_enabled,
            scoring_config=self._sc
        )

    def apply_boost(
        self,
        confidence: float,
        video_segment: SRTSegment,
        boost_type: str
    ) -> Tuple[float, str]:
        """
        Apply a specific boost to confidence score.

        Args:
            confidence: Current confidence
            video_segment: Video segment being scored
            boost_type: Type of boost ('broll', 'project', 'entity')

        Returns:
            Tuple of (boosted_confidence, reason)
        """
        if boost_type == 'broll':
            return apply_broll_boost(confidence, video_segment, self.config)
        elif boost_type == 'project':
            return apply_current_project_boost(confidence, video_segment, self.config)
        elif boost_type == 'entity':
            # For entity boost, we need vo_segment too - return unchanged
            logger.debug(f"Entity boost requires vo_segment, skipping for {boost_type}")
            return confidence, ""
        else:
            logger.warning(f"Unknown boost type: {boost_type}")
            return confidence, ""

    def apply_penalty(
        self,
        confidence: float,
        video_segment: SRTSegment,
        penalty_type: str,
        vo_segment: Optional[SRTSegment] = None,
        video_topics: Optional[dict] = None
    ) -> Tuple[float, str]:
        """
        Apply a specific penalty to confidence score.

        Args:
            confidence: Current confidence
            video_segment: Video segment being scored
            penalty_type: Type of penalty ('topic', 'timing', 'caption_quality')
            vo_segment: Optional voiceover segment (required for topic penalty)
            video_topics: Optional video topics dict (required for topic penalty)

        Returns:
            Tuple of (penalized_confidence, reason)
        """
        if penalty_type == 'topic':
            if vo_segment is None:
                return confidence, ""
            chapter_enabled = getattr(self._mc, 'chapter_matching_enabled', False) if self._mc else False
            topic_penalty = getattr(self._mc, 'topic_mismatch_penalty', 0.15) if self._mc else 0.15
            return apply_topic_penalty(
                confidence, vo_segment, video_segment,
                video_topics or {},
                chapter_enabled,
                topic_penalty
            )
        elif penalty_type == 'timing':
            return apply_timing_penalty(confidence, video_segment, self.config)
        elif penalty_type == 'caption_quality':
            return apply_caption_quality_adjustment(confidence, video_segment, self.config)
        else:
            logger.warning(f"Unknown penalty type: {penalty_type}")
            return confidence, ""

    # Minimum confidence floor to prevent cascading multiplicative penalties
    # from reducing confidence to near-zero (US-46-004)
    CONFIDENCE_FLOOR = 0.05

    # Threshold below which over-penalized matches are logged as warnings
    LOW_CONFIDENCE_WARNING_THRESHOLD = 0.15

    @property
    def confidence_floor(self) -> float:
        """Confidence floor from config or class default."""
        return getattr(self._sc, 'confidence_floor', self.CONFIDENCE_FLOOR) if self._sc else self.CONFIDENCE_FLOOR

    @property
    def low_confidence_warning_threshold(self) -> float:
        """Low confidence warning threshold from config or class default."""
        return getattr(self._sc, 'low_confidence_warning_threshold', self.LOW_CONFIDENCE_WARNING_THRESHOLD) if self._sc else self.LOW_CONFIDENCE_WARNING_THRESHOLD

    # Title relevance boost thresholds (US-70-006)
    TITLE_BOOST_1_KEYWORD = 0.03  # 1 keyword match
    TITLE_BOOST_2_KEYWORDS = 0.05  # 2 keyword matches
    TITLE_BOOST_3_PLUS_KEYWORDS = 0.08  # 3+ keyword matches

    # Description relevance boost thresholds (US-73-003)
    DESC_BOOST_1_KEYWORD = 0.02   # 1 keyword match
    DESC_BOOST_2_KEYWORDS = 0.04  # 2 keyword matches
    DESC_BOOST_3_PLUS_KEYWORDS = 0.06  # 3+ keyword matches
    DESC_MAX_CHARS = 200  # Truncate description before keyword extraction

    # Chapter topic match thresholds (US-70-009, updated US-72-007)
    CHAPTER_TOPIC_BOOST_PARTIAL = 0.05      # Partial match: 1-2 shared keywords
    CHAPTER_TOPIC_BOOST_STRONG = 0.10       # Strong match: 3+ shared keywords
    CHAPTER_TOPIC_MISMATCH_PENALTY = -0.05  # Penalty when chapter topic doesn't match

    # Chapter source consistency default (US-70-011)
    DEFAULT_SOURCE_CONSISTENCY_BOOST = 0.03

    # Tag keyword boost (US-72-005): +0.02 per matching tag, capped at +0.08
    TAG_KEYWORD_BOOST_PER_TAG = 0.02
    TAG_KEYWORD_BOOST_CAP = 0.08

    # Chapter coherence penalty (US-71-004)
    CHAPTER_COHERENCE_PENALTY_PER_SOURCE = -0.03  # Penalty per excess source
    CHAPTER_COHERENCE_PENALTY_CAP = -0.10  # Maximum penalty cap

    # Cross-chapter relevance boost default (US-71-005)
    DEFAULT_RELEVANCE_BOOST_WEIGHT = 0.1

    # Listicle consistency boost (US-71-006)
    LISTICLE_CONSISTENCY_BOOST = 0.04

    # Stopwords for title keyword extraction
    _TITLE_STOPWORDS = frozenset({
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
        'can', 'her', 'was', 'one', 'our', 'out', 'has', 'have',
        'been', 'from', 'this', 'that', 'with', 'they', 'what',
        'will', 'there', 'their', 'about', 'would', 'which', 'into',
        'how', 'why', 'who', 'when', 'where', 'does', 'did', 'its',
        'than', 'then', 'just', 'more', 'some', 'also', 'very',
    })

    def _extract_keywords(self, text: str) -> set:
        """Extract significant keywords from text (>= 3 chars, not stopwords)."""
        if not text:
            return set()
        words = text.lower().split()
        return {
            w.strip('.,!?:;"\'()[]{}|-')
            for w in words
            if len(w.strip('.,!?:;"\'()[]{}|-')) >= 3
            and w.lower().strip('.,!?:;"\'()[]{}|-') not in self._TITLE_STOPWORDS
        }

    def apply_title_relevance_adjustment(
        self,
        confidence: float,
        vo_segment: SRTSegment,
        video_title: Optional[str] = None,
    ) -> Tuple[float, str]:
        """
        Apply title keyword overlap boost to confidence score.

        Extracts keywords from both the voiceover segment text and the video title,
        then applies a graduated boost based on overlap count.

        Args:
            confidence: Current confidence score
            vo_segment: Voiceover segment with text
            video_title: Video title string

        Returns:
            Tuple of (adjusted_confidence, reason)
        """
        if not video_title:
            return confidence, ""

        vo_keywords = self._extract_keywords(vo_segment.text)
        title_keywords = self._extract_keywords(video_title)

        if not vo_keywords or not title_keywords:
            return confidence, ""

        overlap = vo_keywords & title_keywords
        match_count = len(overlap)

        if match_count == 0:
            return confidence, ""

        if match_count >= 3:
            boost = self.TITLE_BOOST_3_PLUS_KEYWORDS
        elif match_count == 2:
            boost = self.TITLE_BOOST_2_KEYWORDS
        else:
            boost = self.TITLE_BOOST_1_KEYWORD

        matched_words = ', '.join(sorted(overlap)[:5])
        reason = f"title relevance boost +{boost} ({match_count} keyword{'s' if match_count != 1 else ''}: {matched_words})"
        return confidence + boost, reason

    def apply_description_relevance(
        self,
        confidence: float,
        vo_segment: SRTSegment,
        video_description: Optional[str] = None,
    ) -> Tuple[float, str]:
        """
        Apply description keyword overlap boost to confidence score (US-73-003).

        Extracts keywords from the voiceover segment text and the first 200 chars
        of the video description, then applies a graduated boost based on overlap.

        Args:
            confidence: Current confidence score
            vo_segment: Voiceover segment with text
            video_description: Video description string

        Returns:
            Tuple of (adjusted_confidence, reason)
        """
        if not video_description:
            return confidence, ""

        truncated = video_description[:self.DESC_MAX_CHARS]
        vo_keywords = self._extract_keywords(vo_segment.text)
        desc_keywords = self._extract_keywords(truncated)

        if not vo_keywords or not desc_keywords:
            return confidence, ""

        overlap = vo_keywords & desc_keywords
        match_count = len(overlap)

        if match_count == 0:
            return confidence, ""

        if match_count >= 3:
            boost = self.DESC_BOOST_3_PLUS_KEYWORDS
        elif match_count == 2:
            boost = self.DESC_BOOST_2_KEYWORDS
        else:
            boost = self.DESC_BOOST_1_KEYWORD

        matched_words = ', '.join(sorted(overlap)[:5])
        reason = f"description relevance boost +{boost} ({match_count} keyword{'s' if match_count != 1 else ''}: {matched_words})"
        return confidence + boost, reason

    def apply_chapter_topic_match(
        self,
        confidence: float,
        vo_segment: SRTSegment,
        chapter_title: Optional[str] = None,
        vo_chapter_index: Optional[int] = None,
    ) -> Tuple[float, str]:
        """
        Apply chapter topic match adjustment to confidence score (US-72-007).

        Compares voiceover chapter keywords against video chapter title keywords.
        Boosts confidence when topics match, applies small penalty on mismatch.
        No adjustment when either side lacks chapter data.

        Thresholds:
        - Partial match (1-2 shared keywords): +0.05
        - Strong match (3+ shared keywords): +0.10
        - Mismatch (0 shared keywords): -0.05

        Args:
            confidence: Current confidence score
            vo_segment: Voiceover segment with text
            chapter_title: Chapter title for the matched video segment
            vo_chapter_index: Voiceover chapter index (None = no chapter data)

        Returns:
            Tuple of (adjusted_confidence, reason)
        """
        # No adjustment when either side lacks chapter data
        if vo_chapter_index is None:
            return confidence, ""

        if not chapter_title:
            return confidence, ""

        vo_keywords = self._extract_keywords(vo_segment.text)
        chapter_keywords = self._extract_keywords(chapter_title)

        if not vo_keywords or not chapter_keywords:
            return confidence, ""

        overlap = vo_keywords & chapter_keywords
        match_count = len(overlap)

        if match_count >= 3:
            adjustment = self.CHAPTER_TOPIC_BOOST_STRONG
            matched_words = ', '.join(sorted(overlap)[:5])
            reason = f"chapter topic strong match +{adjustment} ({match_count} keywords: {matched_words})"
        elif match_count >= 1:
            adjustment = self.CHAPTER_TOPIC_BOOST_PARTIAL
            matched_words = ', '.join(sorted(overlap)[:5])
            reason = f"chapter topic partial match +{adjustment} ({match_count} keyword{'s' if match_count != 1 else ''}: {matched_words})"
        else:
            adjustment = self.CHAPTER_TOPIC_MISMATCH_PENALTY
            reason = f"chapter topic mismatch {adjustment}"

        return confidence + adjustment, reason

    def apply_tag_keyword_boost(
        self,
        confidence: float,
        vo_segment: SRTSegment,
        video_tags: Optional[List[str]] = None,
    ) -> Tuple[float, str]:
        """
        Apply tag-based keyword boost to confidence score (US-71-003).

        Compares voiceover segment keywords against video tags extracted
        during caption fetching. Applies graduated boost based on overlap.

        Args:
            confidence: Current confidence score
            vo_segment: Voiceover segment with text
            video_tags: List of video tags/keywords from CaptionResult

        Returns:
            Tuple of (adjusted_confidence, reason)
        """
        if not video_tags:
            return confidence, ""

        vo_keywords = self._extract_keywords(vo_segment.text)
        if not vo_keywords:
            return confidence, ""

        # Normalize tags to lowercase keyword set
        tag_keywords = {t.lower().strip() for t in video_tags if t and len(t.strip()) >= 3}
        if not tag_keywords:
            return confidence, ""

        overlap = vo_keywords & tag_keywords
        match_count = len(overlap)

        if match_count == 0:
            return confidence, ""

        boost = min(match_count * self.TAG_KEYWORD_BOOST_PER_TAG, self.TAG_KEYWORD_BOOST_CAP)

        matched_words = ', '.join(sorted(overlap)[:5])
        reason = f"tag keyword boost +{boost} ({match_count} tag{'s' if match_count != 1 else ''}: {matched_words})"
        return confidence + boost, reason

    def apply_chapter_source_consistency(
        self,
        confidence: float,
        video_segment: SRTSegment,
        recent_matches: List['Match'],
        current_chapter_index: int = -1,
        segment_chapter_map: Optional[dict] = None,
    ) -> Tuple[float, str]:
        """
        Apply source consistency boost within the same voiceover chapter (US-70-011).

        When a candidate video source matches the previous segment's source AND
        both segments are in the same voiceover chapter, apply a small boost.
        Within a coherent chapter about one topic, reusing the same source is
        desirable for visual continuity.

        Args:
            confidence: Current confidence score
            video_segment: Video segment being considered
            recent_matches: List of recent Match objects (most recent first)
            current_chapter_index: Chapter index for current voiceover segment (-1 = none)
            segment_chapter_map: Optional dict mapping segment index to chapter index

        Returns:
            Tuple of (adjusted_confidence, reason)
        """
        # Only applies when chapter grouping is enabled and we have chapter info
        cg = getattr(self._mc, 'chapter_grouping', None) if self._mc else None
        if cg is None or not getattr(cg, 'enabled', True):
            return confidence, ""

        if current_chapter_index < 0 or not recent_matches:
            return confidence, ""

        current_source = getattr(video_segment, 'source_file', None)
        if not current_source:
            return confidence, ""

        # Check the most recent match
        prev_match = recent_matches[0]
        if prev_match is None:
            return confidence, ""

        prev_source = getattr(prev_match.video_segment, 'source_file', None) if prev_match.video_segment else None
        if prev_source != current_source:
            return confidence, ""

        # Check if previous segment is in the same chapter
        prev_chapter_index = -1
        if segment_chapter_map is not None:
            prev_seg_idx = getattr(prev_match.voiceover_segment, 'index', None) if prev_match.voiceover_segment else None
            if prev_seg_idx is not None:
                prev_chapter_index = segment_chapter_map.get(prev_seg_idx, -1)
        # Also support chapter_index attribute on the voiceover segment itself
        if prev_chapter_index < 0 and prev_match.voiceover_segment:
            prev_chapter_index = getattr(prev_match.voiceover_segment, 'chapter_index', -1)

        if prev_chapter_index != current_chapter_index or prev_chapter_index < 0:
            return confidence, ""

        boost = getattr(cg, 'source_consistency_boost', self.DEFAULT_SOURCE_CONSISTENCY_BOOST)
        reason = f"chapter_source_consistency: +{boost:.2f} (same source in chapter {current_chapter_index})"

        logger.debug(
            f"US-70-011 chapter source consistency: source={current_source}, "
            f"chapter={current_chapter_index}, boost={boost:.2f}"
        )

        return confidence + boost, reason

    def apply_chapter_coherence_penalty(
        self,
        confidence: float,
        current_chapter_index: int,
        chapter_source_counts: Optional[dict] = None,
    ) -> Tuple[float, str]:
        """
        Apply coherence penalty when a voiceover chapter uses too many different video sources (US-71-004).

        When segments in the same chapter are sourced from many different videos,
        it creates a scattered viewing experience. This penalty discourages
        excessive source diversity within a single chapter.

        Args:
            confidence: Current confidence score
            current_chapter_index: Chapter index for current voiceover segment (-1 = none)
            chapter_source_counts: Dict mapping chapter_index -> set of unique source video IDs

        Returns:
            Tuple of (adjusted_confidence, reason)
        """
        if current_chapter_index is None or current_chapter_index < 0 or not chapter_source_counts:
            return confidence, ""

        cg = getattr(self._mc, 'chapter_grouping', None) if self._mc else None
        if cg is None or not getattr(cg, 'enabled', True):
            return confidence, ""

        sources = chapter_source_counts.get(current_chapter_index)
        if not sources:
            return confidence, ""

        threshold = getattr(cg, 'coherence_penalty_threshold', 5)
        source_count = len(sources)
        excess = source_count - threshold

        if excess <= 0:
            return confidence, ""

        penalty = max(
            self.CHAPTER_COHERENCE_PENALTY_CAP,
            excess * self.CHAPTER_COHERENCE_PENALTY_PER_SOURCE,
        )

        reason = (
            f"chapter_coherence: {penalty:.2f} "
            f"({source_count} sources in chapter {current_chapter_index}, threshold {threshold})"
        )

        logger.debug(
            f"US-71-004 chapter coherence penalty: chapter={current_chapter_index}, "
            f"sources={source_count}, threshold={threshold}, penalty={penalty:.2f}"
        )

        return confidence + penalty, reason

    def apply_cross_chapter_relevance_boost(
        self,
        confidence: float,
        current_chapter_index: int,
        video_chapter_index: int,
        relevance_matrix: Optional[List[List[float]]] = None,
    ) -> Tuple[float, str]:
        """
        Apply a soft boost for candidates from high-relevance video chapters (US-71-005).

        When a relevance matrix is available (computed from voiceover x video chapter
        keyword overlap), candidates from video chapters with high topic relevance
        to the current voiceover chapter get a proportional boost.

        Boost = relevance_score * relevance_boost_weight

        Args:
            confidence: Current confidence score
            current_chapter_index: Voiceover chapter index (-1 = none)
            video_chapter_index: Video chapter index (-1 = none)
            relevance_matrix: 2D list [vo_chapter][vid_chapter] of relevance scores (0-1)

        Returns:
            Tuple of (adjusted_confidence, reason)
        """
        if current_chapter_index < 0 or video_chapter_index < 0:
            return confidence, ""

        if not relevance_matrix:
            return confidence, ""

        cg = getattr(self._mc, 'chapter_grouping', None) if self._mc else None
        if cg is None or not getattr(cg, 'enabled', True):
            return confidence, ""

        # Bounds check
        if current_chapter_index >= len(relevance_matrix):
            return confidence, ""
        row = relevance_matrix[current_chapter_index]
        if video_chapter_index >= len(row):
            return confidence, ""

        relevance_score = row[video_chapter_index]
        if relevance_score <= 0.0:
            return confidence, ""

        weight = getattr(cg, 'relevance_boost_weight', self.DEFAULT_RELEVANCE_BOOST_WEIGHT)
        boost = relevance_score * weight

        reason = (
            f"cross_chapter_relevance: +{boost:.3f} "
            f"(vo_ch={current_chapter_index}, vid_ch={video_chapter_index}, "
            f"relevance={relevance_score:.2f}, weight={weight})"
        )

        logger.debug(
            "US-71-005 cross-chapter relevance boost: vo_ch=%d, vid_ch=%d, "
            "relevance=%.2f, weight=%.2f, boost=%.3f",
            current_chapter_index, video_chapter_index, relevance_score, weight, boost,
        )

        return confidence + boost, reason

    def apply_listicle_consistency_boost(
        self,
        confidence: float,
        vo_segment: SRTSegment,
        video_segment: SRTSegment,
        listicle_groups: List[Any],
        recent_matches: Optional[List['Match']] = None,
    ) -> Tuple[float, str]:
        """
        Apply consistency boost for segments within the same listicle group (US-71-006).

        When consecutive segments within the same listicle group match from the same
        video source, apply a small boost to encourage source consistency within
        list items. Segments at group boundaries (first segment of a new group)
        get no boost.

        Args:
            confidence: Current confidence score
            vo_segment: Current voiceover segment
            video_segment: Candidate video segment
            listicle_groups: List of ListicleGroup objects from listicle detection
            recent_matches: Optional list of recent Match objects (most recent first)

        Returns:
            Tuple of (adjusted_confidence, reason)
        """
        if not listicle_groups or not recent_matches:
            return confidence, ""

        # Find which listicle group this voiceover segment belongs to
        seg_idx = getattr(vo_segment, 'index', -1)
        if seg_idx < 0:
            return confidence, ""

        current_group = None
        for group in listicle_groups:
            start = group.start_segment_idx if not isinstance(group, dict) else group.get('start_segment_idx', -1)
            end = group.end_segment_idx if not isinstance(group, dict) else group.get('end_segment_idx', -1)
            if start <= seg_idx <= end:
                current_group = group
                break

        if current_group is None:
            return confidence, ""

        # Check if this is a boundary segment (first segment of the group)
        group_start = current_group.start_segment_idx if not isinstance(current_group, dict) else current_group.get('start_segment_idx', -1)
        if seg_idx == group_start:
            return confidence, ""

        # Check if previous match is in the same group and from the same source
        prev_match = recent_matches[0]
        if prev_match is None or prev_match.video_segment is None:
            return confidence, ""

        prev_seg_idx = getattr(prev_match.voiceover_segment, 'index', -1) if prev_match.voiceover_segment else -1
        if prev_seg_idx < 0:
            return confidence, ""

        # Previous segment must also be in the same listicle group
        prev_in_group = group_start <= prev_seg_idx <= (current_group.end_segment_idx if not isinstance(current_group, dict) else current_group.get('end_segment_idx', -1))
        if not prev_in_group:
            return confidence, ""

        # Check source match
        prev_source = getattr(prev_match.video_segment, 'source_file', None)
        current_source = getattr(video_segment, 'source_file', None)

        if not prev_source or not current_source or prev_source != current_source:
            return confidence, ""

        boost = self.LISTICLE_CONSISTENCY_BOOST
        group_id = current_group.group_id if not isinstance(current_group, dict) else current_group.get('group_id', '?')

        reason = (
            f"listicle_consistency: +{boost:.2f} "
            f"(same source in listicle group {group_id})"
        )

        logger.debug(
            "US-71-006 listicle consistency boost: seg=%d, group=%s, source=%s, boost=%.2f",
            seg_idx, group_id, current_source, boost,
        )

        return confidence + boost, reason

    def is_within_chapter(
        self,
        current_chapter_index: int,
        recent_matches: List['Match'],
        segment_chapter_map: Optional[dict] = None,
    ) -> bool:
        """
        Check if the current segment and previous segment are in the same chapter.

        Used to suppress consecutive_source_penalty within chapter boundaries.

        Args:
            current_chapter_index: Chapter index for current segment (-1 = none)
            recent_matches: Recent matches (most recent first)
            segment_chapter_map: Optional dict mapping segment index to chapter index

        Returns:
            True if both are in the same chapter (and chapter grouping is enabled)
        """
        cg = getattr(self._mc, 'chapter_grouping', None) if self._mc else None
        if cg is None or not getattr(cg, 'enabled', True):
            return False

        if current_chapter_index < 0 or not recent_matches:
            return False

        prev_match = recent_matches[0]
        if prev_match is None:
            return False

        prev_chapter_index = -1
        if segment_chapter_map is not None:
            prev_seg_idx = getattr(prev_match.voiceover_segment, 'index', None) if prev_match.voiceover_segment else None
            if prev_seg_idx is not None:
                prev_chapter_index = segment_chapter_map.get(prev_seg_idx, -1)
        if prev_chapter_index < 0 and prev_match.voiceover_segment:
            prev_chapter_index = getattr(prev_match.voiceover_segment, 'chapter_index', -1)

        return prev_chapter_index == current_chapter_index and prev_chapter_index >= 0

    def apply_all_adjustments(
        self,
        confidence: float,
        vo_segment: SRTSegment,
        video_segment: SRTSegment,
        video_topics: Optional[dict] = None,
        chapter_matching_enabled: bool = False,
        topic_mismatch_penalty: float = 0.15,
        video_title: Optional[str] = None,
        chapter_title: Optional[str] = None,
        recent_matches: Optional[List['Match']] = None,
        current_chapter_index: int = -1,
        segment_chapter_map: Optional[dict] = None,
        video_tags: Optional[List[str]] = None,
        chapter_source_counts: Optional[dict] = None,
        relevance_matrix: Optional[List[List[float]]] = None,
        video_chapter_index: int = -1,
        listicle_groups: Optional[List[Any]] = None,
        video_description: Optional[str] = None,
    ) -> Tuple[float, str, list]:
        """
        Apply all scoring adjustments in the correct order.

        Order: topic_penalty -> broll_boost -> caption_quality -> timing_penalty
               -> project_boost -> title_relevance -> description_relevance
               -> chapter_topic_match
               -> chapter_source_consistency -> tag_keyword_boost
               -> chapter_coherence_penalty -> cross_chapter_relevance
               -> listicle_consistency
        After all adjustments, a minimum confidence floor is enforced to prevent
        cascading multiplicative penalties from reducing confidence to near-zero.

        Args:
            confidence: Base confidence score
            vo_segment: Voiceover segment
            video_segment: Video segment
            video_topics: Optional dict of video topics
            chapter_matching_enabled: Whether chapter matching is enabled
            topic_mismatch_penalty: Maximum topic mismatch penalty
            video_title: Optional video title for title relevance scoring
            chapter_title: Optional chapter title for chapter topic matching
            recent_matches: Optional list of recent Match objects for source consistency
            current_chapter_index: Chapter index for current voiceover segment (-1 = none)
            segment_chapter_map: Optional dict mapping segment index to chapter index
            video_tags: Optional list of video tags for tag keyword boosting
            chapter_source_counts: Optional dict mapping chapter_index -> set of unique source IDs
            relevance_matrix: Optional cross-chapter relevance matrix (US-71-005)
            video_chapter_index: Video chapter index for relevance lookup (-1 = none)
            listicle_groups: Optional list of ListicleGroup objects for within-group consistency (US-71-006)

        Returns:
            Tuple of (adjusted_confidence, combined_reason, confidence_breakdown)
            where confidence_breakdown is a list of dicts with keys: component, adjustment, reason
        """
        original_confidence = confidence
        reasons = []
        breakdown = []

        # 1. Topic penalty
        prev = confidence
        confidence, topic_reason = apply_topic_penalty(
            confidence, vo_segment, video_segment,
            video_topics or {},
            chapter_matching_enabled,
            topic_mismatch_penalty
        )
        if topic_reason:
            reasons.append(topic_reason)
            breakdown.append({'component': 'topic_penalty', 'adjustment': round(confidence - prev, 4), 'reason': topic_reason})

        # 2. B-roll boost
        prev = confidence
        confidence, broll_reason = apply_broll_boost(
            confidence, video_segment, self.config
        )
        if broll_reason:
            reasons.append(broll_reason)
            breakdown.append({'component': 'broll_boost', 'adjustment': round(confidence - prev, 4), 'reason': broll_reason})

        # 3. Caption quality adjustment
        prev = confidence
        confidence, caption_reason = apply_caption_quality_adjustment(
            confidence, video_segment, self.config
        )
        if caption_reason:
            reasons.append(caption_reason)
            breakdown.append({'component': 'caption_quality', 'adjustment': round(confidence - prev, 4), 'reason': caption_reason})

        # 3b. Tiered caption quality penalties (US-73-006)
        prev = confidence
        confidence, tiered_entries = apply_tiered_caption_penalties(
            confidence, video_segment, self.config
        )
        if tiered_entries:
            for entry in tiered_entries:
                reasons.append(entry['reason'])
                breakdown.append(entry)

        # 4. Timing penalty
        prev = confidence
        confidence, timing_reason = apply_timing_penalty(
            confidence, video_segment, self.config
        )
        if timing_reason:
            reasons.append(timing_reason)
            breakdown.append({'component': 'timing_penalty', 'adjustment': round(confidence - prev, 4), 'reason': timing_reason})

        # 5. Project boost (global cache penalty)
        prev = confidence
        confidence, project_reason = apply_current_project_boost(
            confidence, video_segment, self.config
        )
        if project_reason:
            reasons.append(project_reason)
            breakdown.append({'component': 'project_boost', 'adjustment': round(confidence - prev, 4), 'reason': project_reason})

        # 6. Title relevance boost (US-70-006)
        if video_title:
            prev = confidence
            confidence, title_reason = self.apply_title_relevance_adjustment(
                confidence, vo_segment, video_title
            )
            if title_reason:
                reasons.append(title_reason)
                breakdown.append({'component': 'title_relevance', 'adjustment': round(confidence - prev, 4), 'reason': title_reason})

        # 6b. Description relevance boost (US-73-003)
        if video_description:
            prev = confidence
            confidence, desc_reason = self.apply_description_relevance(
                confidence, vo_segment, video_description
            )
            if desc_reason:
                reasons.append(desc_reason)
                breakdown.append({'component': 'description_relevance', 'adjustment': round(confidence - prev, 4), 'reason': desc_reason})

        # 7. Chapter topic match (US-70-009, US-72-007)
        if chapter_title:
            vo_ch_idx = current_chapter_index if current_chapter_index >= 0 else None
            prev = confidence
            confidence, chapter_reason = self.apply_chapter_topic_match(
                confidence, vo_segment, chapter_title, vo_chapter_index=vo_ch_idx
            )
            if chapter_reason:
                reasons.append(chapter_reason)
                breakdown.append({'component': 'chapter_topic_match', 'adjustment': round(confidence - prev, 4), 'reason': chapter_reason})

        # 8. Chapter source consistency boost (US-70-011)
        if recent_matches is not None:
            prev = confidence
            confidence, consistency_reason = self.apply_chapter_source_consistency(
                confidence, video_segment, recent_matches,
                current_chapter_index, segment_chapter_map
            )
            if consistency_reason:
                reasons.append(consistency_reason)
                breakdown.append({'component': 'chapter_source_consistency', 'adjustment': round(confidence - prev, 4), 'reason': consistency_reason})

        # 9. Tag keyword boost (US-71-003)
        if video_tags:
            prev = confidence
            confidence, tag_reason = self.apply_tag_keyword_boost(
                confidence, vo_segment, video_tags
            )
            if tag_reason:
                reasons.append(tag_reason)
                breakdown.append({'component': 'tag_keyword_boost', 'adjustment': round(confidence - prev, 4), 'reason': tag_reason})

        # 10. Chapter coherence penalty (US-71-004)
        if chapter_source_counts is not None:
            prev = confidence
            confidence, coherence_reason = self.apply_chapter_coherence_penalty(
                confidence, current_chapter_index, chapter_source_counts
            )
            if coherence_reason:
                reasons.append(coherence_reason)
                breakdown.append({'component': 'chapter_coherence_penalty', 'adjustment': round(confidence - prev, 4), 'reason': coherence_reason})

        # 11. Cross-chapter relevance boost (US-71-005)
        if relevance_matrix and current_chapter_index >= 0 and video_chapter_index >= 0:
            prev = confidence
            confidence, relevance_reason = self.apply_cross_chapter_relevance_boost(
                confidence, current_chapter_index, video_chapter_index, relevance_matrix
            )
            if relevance_reason:
                reasons.append(relevance_reason)
                breakdown.append({'component': 'cross_chapter_relevance', 'adjustment': round(confidence - prev, 4), 'reason': relevance_reason})

        # 12. Listicle consistency boost (US-71-006)
        if listicle_groups:
            prev = confidence
            confidence, listicle_reason = self.apply_listicle_consistency_boost(
                confidence, vo_segment, video_segment, listicle_groups, recent_matches
            )
            if listicle_reason:
                reasons.append(listicle_reason)
                breakdown.append({'component': 'listicle_consistency', 'adjustment': round(confidence - prev, 4), 'reason': listicle_reason})

        # 13. Enforce minimum confidence floor (US-46-004)
        # Prevents cascading multiplicative penalties from reducing confidence to near-zero
        floor = self.confidence_floor
        if confidence < floor and original_confidence > floor:
            breakdown.append({'component': 'confidence_floor', 'adjustment': round(floor - confidence, 4), 'reason': f'confidence floor applied: {floor}'})
            confidence = floor
            reasons.append(f"confidence floor applied: {floor}")

        # 14. Log warning for over-penalized matches (US-46-004)
        warning_threshold = self.low_confidence_warning_threshold
        if confidence < warning_threshold and original_confidence >= warning_threshold:
            logger.warning(
                f"Over-penalized match: {original_confidence:.2f} -> {confidence:.2f} "
                f"(video={getattr(video_segment, 'source_file', 'unknown')}, "
                f"adjustments: {' | '.join(reasons)})"
            )

        combined_reason = " | ".join(reasons) if reasons else ""

        # 15. Log confidence breakdown at DEBUG level (US-53-003)
        if breakdown:
            parts = [f"{b['component']}: {b['adjustment']:+.2f}" for b in breakdown]
            logger.debug(
                f"confidence: {original_confidence:.2f} -> {confidence:.2f} ({', '.join(parts)})"
            )

        return confidence, combined_reason, breakdown

    def calculate_adaptive_threshold(
        self,
        base_threshold: float,
        voiceover_text: str,
        candidates: List[Tuple[SRTSegment, float]]
    ) -> Tuple[float, str]:
        """
        Calculate adaptive LLM skip threshold.

        Args:
            base_threshold: Base skip_llm_threshold
            voiceover_text: Voiceover segment text
            candidates: Candidate list

        Returns:
            Tuple of (adjusted_threshold, reason)
        """
        return calculate_adaptive_threshold(
            base_threshold,
            voiceover_text,
            candidates,
            self.config
        )

    def extract_entity_texts(self, segment: SRTSegment) -> List[str]:
        """
        Extract entity texts from a segment.

        Args:
            segment: Segment to extract entities from

        Returns:
            List of entity text strings
        """
        return _extract_entity_texts(segment)

    def calculate_keyword_overlap(
        self,
        vo_keywords: List[str],
        video_keywords: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Calculate keyword overlap score.

        Args:
            vo_keywords: Voiceover keywords
            video_keywords: Video keywords

        Returns:
            Tuple of (overlap_score, matched_keywords)
        """
        return calculate_keyword_overlap_score(vo_keywords, video_keywords, scoring_config=self._sc)

    def calculate_entity_match(
        self,
        vo_entities: List[str],
        video_entities: List[str]
    ) -> Tuple[float, List[str]]:
        """
        Calculate entity match score.

        Args:
            vo_entities: Voiceover entities
            video_entities: Video entities

        Returns:
            Tuple of (entity_score, matched_entities)
        """
        return calculate_entity_match_score(vo_entities, video_entities)


def normalize_confidence_by_pool(
    confidence: float,
    pool_size: int,
    candidates: Optional[List[Tuple[SRTSegment, float]]] = None,
    pool_normalization_enabled: bool = True,
    scoring_config=None
) -> Tuple[float, str]:
    """
    Normalize confidence score based on candidate pool size.

    Confidence scores are inherently relative to the pool size:
    - Small pools (<10): High confidence is meaningful when top match is clear
    - Medium pools (10-100): Standard interpretation
    - Large pools (>100): High confidence may be overfit, tight margins are concerning

    Normalization formula: sqrt(pool_size / 50), capped to [0.8, 1.2]

    Additional adjustments:
    - Small pool with clear winner (top > 2nd by 0.1+): Boost confidence
    - Large pool with tight margin (top-2nd < 0.05): Reduce confidence

    Args:
        confidence: Original confidence score (0.0 - 1.0)
        pool_size: Number of candidates in the pool
        candidates: Optional list of (segment, similarity) tuples sorted by similarity descending
                   Used to determine if top match is clear vs tight margins
        pool_normalization_enabled: Whether to apply pool normalization (config option)

    Returns:
        Tuple of (normalized_confidence, reason_string)
    """
    if not pool_normalization_enabled:
        return confidence, "pool_normalization_disabled"

    if pool_size <= 0:
        return confidence, "empty_pool"

    # Handle NaN/Inf confidence input
    import math
    if math.isnan(confidence):
        return 0.0, "nan_confidence_input"
    if math.isinf(confidence):
        return 1.0 if confidence > 0 else 0.0, "inf_confidence_input"

    reasons = []
    adjustment = 0.0

    # Read configurable pool normalization constants (fall back to module constants)
    ref_size = getattr(scoring_config, 'pool_normalization_reference_size', POOL_NORMALIZATION_REFERENCE_SIZE) if scoring_config else POOL_NORMALIZATION_REFERENCE_SIZE
    min_factor = getattr(scoring_config, 'pool_normalization_min_factor', POOL_NORMALIZATION_MIN_FACTOR) if scoring_config else POOL_NORMALIZATION_MIN_FACTOR
    max_factor = getattr(scoring_config, 'pool_normalization_max_factor', POOL_NORMALIZATION_MAX_FACTOR) if scoring_config else POOL_NORMALIZATION_MAX_FACTOR
    small_threshold = getattr(scoring_config, 'pool_small_threshold', POOL_SMALL_THRESHOLD) if scoring_config else POOL_SMALL_THRESHOLD
    large_threshold = getattr(scoring_config, 'pool_large_threshold', POOL_LARGE_THRESHOLD) if scoring_config else POOL_LARGE_THRESHOLD
    tight_margin = getattr(scoring_config, 'pool_tight_margin_threshold', POOL_TIGHT_MARGIN_THRESHOLD) if scoring_config else POOL_TIGHT_MARGIN_THRESHOLD

    # Calculate base normalization factor: sqrt(pool_size / reference_size)
    # Small pools: factor < 1.0 (confidence preserved/boosted)
    # Large pools: factor > 1.0 (confidence reduced)
    raw_factor = (pool_size / ref_size) ** 0.5

    # Clamp factor to [min_factor, max_factor] range
    clamped_factor = max(min_factor, min(max_factor, raw_factor))

    # Determine margin characteristics if candidates provided
    top_margin = None
    if candidates and len(candidates) >= 2:
        top_score = candidates[0][1]
        second_score = candidates[1][1]
        top_margin = top_score - second_score

    # Apply pool-specific adjustments
    if pool_size < small_threshold:
        # Small pool: boost confidence if there's a clear winner
        if top_margin is not None and top_margin >= 0.1:
            # Clear winner in small pool - this is a strong signal
            adjustment = 0.05
            reasons.append(f"small_pool({pool_size})+clear_winner({top_margin:.2f}):+0.05")
        else:
            reasons.append(f"small_pool({pool_size}):factor={clamped_factor:.2f}")

    elif pool_size > large_threshold:
        # Large pool: reduce confidence if margins are tight
        if top_margin is not None and top_margin < tight_margin:
            # Tight margin in large pool - confidence may be inflated
            adjustment = -0.05
            reasons.append(f"large_pool({pool_size})+tight_margin({top_margin:.2f}):-0.05")
        else:
            reasons.append(f"large_pool({pool_size}):factor={clamped_factor:.2f}")

    else:
        # Medium pool - use factor-based adjustment
        reasons.append(f"medium_pool({pool_size}):factor={clamped_factor:.2f}")

    # Apply normalization: higher factor = lower confidence
    # Invert the factor logic: use 1/factor for confidence adjustment
    # Small pool (factor 0.8) -> 1/0.8 = 1.25 multiplier (boost)
    # Large pool (factor 1.2) -> 1/1.2 = 0.83 multiplier (reduce)
    inverse_factor = 1.0 / clamped_factor

    # Calculate normalized confidence
    normalized = confidence * inverse_factor + adjustment

    # Clamp to valid range [0.0, 1.0]
    normalized = max(0.0, min(1.0, normalized))

    reason = "; ".join(reasons)

    # Determine reason category for transparency logging
    if pool_size < small_threshold and top_margin is not None and top_margin >= 0.1:
        reason_category = "small_pool_boost"
    elif pool_size > large_threshold and top_margin is not None and top_margin < tight_margin:
        reason_category = "large_pool_penalty"
    else:
        reason_category = "no_adjustment"

    logger.debug(
        f"Pool normalization: pool_size={pool_size}, top_margin={f'{top_margin:.4f}' if top_margin is not None else 'N/A'}, "
        f"factor={clamped_factor:.3f}, reason_category={reason_category}, "
        f"{confidence:.3f} -> {normalized:.3f} ({reason})"
    )

    return normalized, reason


def apply_consecutive_source_penalty(
    confidence: float,
    video_segment: SRTSegment,
    recent_matches: List['Match'],
    config=None,
    suppress_in_chapter: bool = False,
) -> Tuple[float, str]:
    """
    Apply penalty for using the same video source in consecutive segments (US-63-009).

    Visual variety is important in the final edit. Using the same video source
    repeatedly in adjacent segments creates a monotonous viewing experience.
    This function applies a stacking penalty for consecutive same-source matches.

    When suppress_in_chapter is True (US-70-011), the penalty is suppressed because
    within a coherent chapter, source consistency is desirable.

    Args:
        confidence: Current confidence score
        video_segment: Video segment being considered
        recent_matches: List of recent Match objects (most recent first), used to check
                       if previous N matches used the same source
        config: Optional config object with matching settings
        suppress_in_chapter: If True, skip penalty (segments in same chapter)

    Returns:
        Tuple of (adjusted_confidence, reason_string)
    """
    if not recent_matches:
        return confidence, ""

    # US-70-011: Suppress penalty within chapter boundaries
    if suppress_in_chapter:
        return confidence, ""

    # Get config values
    if config is not None:
        mc = getattr(config, 'matching', None)
        if mc is not None:
            penalty_per_consecutive = getattr(mc, 'consecutive_source_penalty', 0.1)
            max_consecutive = getattr(mc, 'max_consecutive_same_source', 3)
        else:
            penalty_per_consecutive = 0.1
            max_consecutive = 3
    else:
        penalty_per_consecutive = 0.1
        max_consecutive = 3

    # Get current video source
    current_source = getattr(video_segment, 'source_file', None)
    if not current_source:
        return confidence, ""

    # Count consecutive same-source matches
    consecutive_count = 0
    for match in recent_matches:
        if match is None:
            break
        match_source = getattr(match.video_segment, 'source_file', None) if match.video_segment else None
        if match_source == current_source:
            consecutive_count += 1
        else:
            break  # Stop counting when we hit a different source

    if consecutive_count == 0:
        return confidence, ""

    # Calculate stacking penalty
    total_penalty = penalty_per_consecutive * consecutive_count

    # Apply penalty
    adjusted = max(0.0, confidence - total_penalty)

    # Build reason string
    reason = f"consecutive_source_penalty: -{total_penalty:.2f} ({consecutive_count} consecutive)"

    logger.debug(
        f"US-63-009 consecutive source penalty: source={current_source}, "
        f"consecutive={consecutive_count}, penalty={total_penalty:.2f}, "
        f"{confidence:.2f} -> {adjusted:.2f}"
    )

    return adjusted, reason


def check_consecutive_source_hard_cap(
    video_segment: SRTSegment,
    recent_matches: List['Match'],
    config=None
) -> Tuple[bool, int]:
    """
    Check if using this video source would exceed the consecutive same-source hard cap.

    Args:
        video_segment: Video segment being considered
        recent_matches: List of recent Match objects (most recent first)
        config: Optional config object with matching settings

    Returns:
        Tuple of (should_block, consecutive_count)
        - should_block: True if this source should be blocked
        - consecutive_count: Number of consecutive matches from this source
    """
    if not recent_matches:
        return False, 0

    # Get config values
    if config is not None:
        mc = getattr(config, 'matching', None)
        if mc is not None:
            max_consecutive = getattr(mc, 'max_consecutive_same_source', 3)
        else:
            max_consecutive = 3
    else:
        max_consecutive = 3

    # Get current video source
    current_source = getattr(video_segment, 'source_file', None)
    if not current_source:
        return False, 0

    # Count consecutive same-source matches
    consecutive_count = 0
    for match in recent_matches:
        if match is None:
            break
        match_source = getattr(match.video_segment, 'source_file', None) if match.video_segment else None
        if match_source == current_source:
            consecutive_count += 1
        else:
            break

    # Block if would exceed hard cap (current use would be consecutive_count + 1)
    should_block = consecutive_count >= max_consecutive

    if should_block:
        logger.debug(
            f"US-63-009 hard cap reached: source={current_source}, "
            f"consecutive={consecutive_count}, max={max_consecutive}"
        )

    return should_block, consecutive_count


def normalize_pool_batch(
    segments: List[Tuple[float, int, Optional[List[Tuple[SRTSegment, float]]]]],
    pool_normalization_enabled: bool = True,
    scoring_config=None
) -> List[Tuple[float, str]]:
    """
    Normalize confidence scores for a batch of segments and log a summary.

    Each entry in segments is a tuple of (confidence, pool_size, candidates).
    After normalizing all segments, logs an INFO-level summary of how many
    received boost vs penalty vs no adjustment.

    Args:
        segments: List of (confidence, pool_size, candidates) tuples
        pool_normalization_enabled: Whether pool normalization is enabled
        scoring_config: Optional MatchingScoringConfig instance

    Returns:
        List of (normalized_confidence, reason) tuples
    """
    results = []
    boost_count = 0
    penalty_count = 0
    no_adj_count = 0

    # Read thresholds for category classification
    small_threshold = getattr(scoring_config, 'pool_small_threshold', POOL_SMALL_THRESHOLD) if scoring_config else POOL_SMALL_THRESHOLD
    large_threshold = getattr(scoring_config, 'pool_large_threshold', POOL_LARGE_THRESHOLD) if scoring_config else POOL_LARGE_THRESHOLD
    tight_margin_val = getattr(scoring_config, 'pool_tight_margin_threshold', POOL_TIGHT_MARGIN_THRESHOLD) if scoring_config else POOL_TIGHT_MARGIN_THRESHOLD

    for confidence, pool_size, candidates in segments:
        normalized, reason = normalize_confidence_by_pool(
            confidence=confidence,
            pool_size=pool_size,
            candidates=candidates,
            pool_normalization_enabled=pool_normalization_enabled,
            scoring_config=scoring_config
        )
        results.append((normalized, reason))

        # Classify for summary
        top_margin = None
        if candidates and len(candidates) >= 2:
            top_margin = candidates[0][1] - candidates[1][1]

        if pool_size < small_threshold and top_margin is not None and top_margin >= 0.1:
            boost_count += 1
        elif pool_size > large_threshold and top_margin is not None and top_margin < tight_margin_val:
            penalty_count += 1
        else:
            no_adj_count += 1

    total = len(segments)
    if total > 0:
        logger.info(
            f"Pool normalization summary: {total} segments processed — "
            f"boost={boost_count}, penalty={penalty_count}, no_adjustment={no_adj_count}"
        )

    return results
