"""
Confidence scoring and adjustment functions.

Migrated from TieredMatcher methods in matching.py (lines 548-960).
Provides confidence adjustments for:
- Duration mismatch penalties
- Topic-based penalties for chapter matching
- B-roll footage boosts
- Current project vs global cache scoring
"""

from typing import Tuple, List, Optional
import logging

from ..utils import SRTSegment
from ..topic_extraction import compute_topic_penalty

logger = logging.getLogger(__name__)


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

    # Compute penalty based on topic mismatch
    penalty = compute_topic_penalty(
        vo_topics=vo_topics,
        video_topics=video_topics_list,
        max_penalty=topic_mismatch_penalty,
        min_overlap=1
    )

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
