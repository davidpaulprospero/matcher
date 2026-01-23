"""
Confidence scoring and adjustment functions.

Migrated from TieredMatcher methods in matching.py (lines 548-960).
Provides confidence adjustments for:
- Duration mismatch penalties
- Topic-based penalties for chapter matching
- B-roll footage boosts
- Current project vs global cache scoring
- Premise-based scoring (video theme/topic matching)
"""

from typing import Tuple, List, Optional, Dict, Any
import logging

from ..utils import SRTSegment
from ..topic_extraction import compute_topic_penalty

logger = logging.getLogger(__name__)


def apply_premise_scoring(
    embedding_similarity: float,
    vo_segment: SRTSegment,
    video_segment: SRTSegment,
    video_premises: Dict[str, str],
    premise_config: Any
) -> Tuple[float, float, str]:
    """
    Apply premise-based scoring to blend theme matching with embedding similarity.

    Premise scoring replaces word-for-word transcript matching with theme-based
    matching. Each video has a premise (e.g., "Documentary about restaurant closures")
    that is compared to the voiceover content.

    New weights (when premise scoring enabled):
    - premise_weight: 50% (video theme ↔ voiceover topic)
    - keyword_weight: 25% (keywords in common)
    - embedding_weight: 15% (semantic similarity - reduced)
    - transcript_weight: 10% (word-for-word - heavily reduced)

    Args:
        embedding_similarity: Original embedding similarity score (0-1)
        vo_segment: Voiceover segment being matched
        video_segment: Video segment being considered
        video_premises: Dict mapping video_id to premise string
        premise_config: PremiseScoringConfig with weights and settings

    Returns:
        Tuple of (blended_score, premise_score, reason)
    """
    if not premise_config or not getattr(premise_config, 'enabled', True):
        return embedding_similarity, 0.0, ""

    # Get video premise
    video_path = video_segment.source_file or ""
    video_id = _extract_video_id_for_premise(video_path)
    premise = video_premises.get(video_id, "")

    if not premise:
        # No premise available - fall back to embedding similarity
        return embedding_similarity, 0.0, "no premise"

    # Get weights from config
    premise_weight = getattr(premise_config, 'premise_weight', 0.50)
    embedding_weight = getattr(premise_config, 'embedding_weight', 0.15)
    keyword_weight = getattr(premise_config, 'keyword_weight', 0.25)
    transcript_weight = getattr(premise_config, 'transcript_weight', 0.10)

    # Compute premise match score using keyword overlap
    vo_text = vo_segment.text.lower() if vo_segment.text else ""
    premise_lower = premise.lower()

    # Simple keyword-based premise matching
    # Extract meaningful words from both texts
    vo_words = set(w for w in vo_text.split() if len(w) > 3)
    premise_words = set(w for w in premise_lower.split() if len(w) > 3)

    if not vo_words or not premise_words:
        return embedding_similarity, 0.0, "insufficient text"

    # Calculate overlap-based premise score
    overlap = vo_words & premise_words
    overlap_ratio = len(overlap) / max(len(vo_words), len(premise_words))

    # Check for content type mismatch (music video, lyrics, etc.)
    content_type_penalty = 0.0
    music_indicators = ['music', 'lyric', 'lyrics', 'song', 'audio', 'karaoke']
    if any(ind in premise_lower for ind in music_indicators):
        # Music content - check if voiceover is about music
        vo_music_related = any(ind in vo_text for ind in music_indicators)
        if not vo_music_related:
            # Music video being matched to non-music content - heavy penalty
            content_type_penalty = 0.5
            logger.debug(f"Premise penalty: music video for non-music content")

    # Compute premise score (0-1)
    premise_score = max(0.0, overlap_ratio - content_type_penalty)

    # Blend scores using configured weights
    # Note: embedding_similarity already includes semantic matching
    # keyword and transcript components are approximated via the overlap
    blended_score = (
        premise_weight * premise_score +
        embedding_weight * embedding_similarity +
        keyword_weight * overlap_ratio +  # Keywords approximated by overlap
        transcript_weight * embedding_similarity  # Transcript approximated by embedding
    )

    # Normalize to 0-1 range
    blended_score = max(0.0, min(1.0, blended_score))

    # Build reason string
    reason = f"premise: {premise_score:.2f}"
    if content_type_penalty > 0:
        reason += f" (music penalty: -{content_type_penalty:.2f})"

    logger.debug(f"Premise scoring: embed={embedding_similarity:.2f}, premise={premise_score:.2f}, blended={blended_score:.2f}")

    return blended_score, premise_score, reason


def _extract_video_id_for_premise(file_path: str) -> str:
    """Extract video ID from file path for premise lookup."""
    import re
    from pathlib import Path

    filename = Path(file_path).stem if file_path else ""

    # Audio-first segment: {video_id}_{offset:04d}
    match = re.match(r'^([a-zA-Z0-9_-]{11})_(\d{4})$', filename)
    if match:
        return match.group(1)

    # Regular YouTube: {title}_{video_id}
    match = re.search(r'_([a-zA-Z0-9_-]{11})$', filename)
    if match:
        return match.group(1)

    # Stock footage - use filename
    if filename.startswith(('pexels_', 'pixabay_')):
        return filename

    # Fallback: use filename as-is
    return filename


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


def apply_caption_boost(
    confidence: float,
    video_segment: SRTSegment,
    config
) -> Tuple[float, str]:
    """
    Apply confidence boost for manual caption transcripts.

    Manual captions (human-created) are typically higher quality than:
    - Auto-generated captions (YouTube's ASR)
    - Whisper transcriptions

    This boost rewards videos with verified human transcriptions.

    Args:
        confidence: Original confidence score
        video_segment: Video segment being considered
        config: Config with download.caption_first.confidence_boost_manual setting

    Returns:
        Tuple of (boosted_confidence, boost_reason)
    """
    # Check transcript source
    transcript_source = getattr(video_segment, 'transcript_source', '')

    # Only boost manual captions
    if transcript_source != 'manual_caption':
        return confidence, ""

    # Get boost amount from config
    download_config = getattr(config, 'download', None)
    caption_config = getattr(download_config, 'caption_first', None) if download_config else None
    caption_boost = getattr(caption_config, 'confidence_boost_manual', 0.1) if caption_config else 0.1

    if caption_boost <= 0:
        return confidence, ""

    boosted = min(1.0, confidence + caption_boost)
    reason = f"manual caption boost: +{caption_boost:.2f}"

    logger.debug(f"Manual caption boost applied: {confidence:.2f} -> {boosted:.2f}")

    return boosted, reason


def apply_chapter_keyword_boost(
    confidence: float,
    video_segment: SRTSegment,
    chapter_keywords: List[str],
    config
) -> Tuple[float, str]:
    """
    Apply confidence boost when video matches chapter keywords.

    When a voiceover segment belongs to a chapter (e.g., "Denny's chapter"),
    videos that contain chapter-related keywords get a confidence boost.

    Checks:
    1. Video transcript contains chapter keywords
    2. Video filename contains chapter keywords
    3. Video source_keyword matches chapter keywords

    Args:
        confidence: Original confidence score
        video_segment: Video segment being considered
        chapter_keywords: Combined keywords from chapter (visual + context + topics + title)
        config: Config with matching.chapter_keyword_boost setting

    Returns:
        Tuple of (boosted_confidence, boost_reason)
    """
    if not chapter_keywords:
        return confidence, ""

    # Get boost settings from config
    mc = getattr(config, 'matching', config)
    keyword_boost_per_match = getattr(mc, 'chapter_keyword_boost', 0.05)
    max_keyword_boost = getattr(mc, 'max_chapter_keyword_boost', 0.15)

    if keyword_boost_per_match <= 0:
        return confidence, ""

    # Get video text content for matching
    video_text = video_segment.text.lower() if video_segment.text else ""
    video_file = video_segment.source_file.lower() if video_segment.source_file else ""
    source_keyword = getattr(video_segment, 'source_keyword', '')
    source_keyword = source_keyword.lower() if source_keyword else ""

    # Extract just the filename without path
    from pathlib import Path
    video_filename = Path(video_file).stem.lower() if video_file else ""

    # Check for keyword matches
    matches = []
    for kw in chapter_keywords:
        if not kw:
            continue
        kw_lower = kw.lower().strip()
        if len(kw_lower) < 2:  # Skip very short keywords
            continue

        # Check transcript, filename, and source keyword
        if kw_lower in video_text or kw_lower in video_filename or kw_lower in source_keyword:
            matches.append(kw)

    if not matches:
        return confidence, ""

    # Calculate boost (capped at max)
    boost = min(max_keyword_boost, len(matches) * keyword_boost_per_match)
    boosted = min(1.0, confidence + boost)

    # Format reason with first few matches
    match_preview = ', '.join(matches[:3])
    if len(matches) > 3:
        match_preview += f" +{len(matches) - 3} more"
    reason = f"chapter keyword boost: +{boost:.2f} ({match_preview})"

    logger.debug(f"Chapter keyword boost applied: {confidence:.2f} -> {boosted:.2f} [{match_preview}]")

    return boosted, reason


def apply_voiceover_keyword_boost(
    confidence: float,
    vo_segment: SRTSegment,
    video_segment: SRTSegment,
    known_entities: List[str],
    config
) -> Tuple[float, str]:
    """
    Apply confidence boost when video matches keywords found in voiceover segment.

    This addresses the "chapter matching" problem differently - instead of relying
    on pre-detected chapters, it extracts key terms (especially proper nouns like
    "Denny's", "Wahlburgers") from the voiceover text and boosts videos that contain them.

    Args:
        confidence: Original confidence score
        vo_segment: Voiceover segment being matched
        video_segment: Video segment being considered
        known_entities: List of entity names extracted from the full voiceover
        config: Config with matching settings

    Returns:
        Tuple of (boosted_confidence, boost_reason)
    """
    if not known_entities:
        return confidence, ""

    # Get boost settings from config
    mc = getattr(config, 'matching', config)
    keyword_boost_per_match = getattr(mc, 'voiceover_keyword_boost', 0.08)
    max_keyword_boost = getattr(mc, 'max_voiceover_keyword_boost', 0.15)

    if keyword_boost_per_match <= 0:
        return confidence, ""

    # Get voiceover text
    vo_text = vo_segment.text.lower() if vo_segment.text else ""
    if not vo_text:
        return confidence, ""

    # Find which entities appear in this voiceover segment
    vo_entities = []
    for entity in known_entities:
        if entity and entity.lower() in vo_text:
            vo_entities.append(entity)

    if not vo_entities:
        return confidence, ""

    # Get video text content for matching
    video_text = video_segment.text.lower() if video_segment.text else ""
    video_file = video_segment.source_file.lower() if video_segment.source_file else ""
    source_keyword = getattr(video_segment, 'source_keyword', '')
    source_keyword = source_keyword.lower() if source_keyword else ""

    # Extract just the filename without path
    from pathlib import Path
    video_filename = Path(video_file).stem.lower() if video_file else ""

    # Check for entity matches in video content
    # Include full path (video_file) since folder names often contain entity names like "Denny_segments"
    matches = []
    for entity in vo_entities:
        entity_lower = entity.lower()
        if (entity_lower in video_text or
            entity_lower in video_file or  # Full path including folder name
            entity_lower in video_filename or
            entity_lower in source_keyword):
            matches.append(entity)

    if not matches:
        return confidence, ""

    # Calculate boost (capped at max)
    boost = min(max_keyword_boost, len(matches) * keyword_boost_per_match)
    boosted = min(1.0, confidence + boost)

    # Format reason with first few matches
    match_preview = ', '.join(matches[:2])
    if len(matches) > 2:
        match_preview += f" +{len(matches) - 2}"
    reason = f"vo-keyword boost: +{boost:.2f} ({match_preview})"

    logger.debug(f"Voiceover keyword boost applied: {confidence:.2f} -> {boosted:.2f} [{match_preview}]")

    return boosted, reason


def apply_entity_mismatch_penalty(
    confidence: float,
    vo_segment: SRTSegment,
    video_segment: SRTSegment,
    known_entities: List[str],
    config
) -> Tuple[float, str]:
    """
    Apply confidence penalty when video is from entity-specific folder but entity not mentioned.

    This is the INVERSE of voiceover_keyword_boost - it penalizes videos that are clearly
    associated with a specific entity (e.g., from "Denny_segments" folder) when that entity
    is NOT mentioned in the current voiceover segment.

    This prevents "Denny's" footage from appearing throughout the timeline - it should
    only appear during the Denny's chapter.

    Args:
        confidence: Original confidence score
        vo_segment: Voiceover segment being matched
        video_segment: Video segment being considered
        known_entities: List of entity names extracted from the full voiceover
        config: Config with matching settings

    Returns:
        Tuple of (adjusted_confidence, penalty_reason)
    """
    if not known_entities:
        return confidence, ""

    # Get penalty settings from config
    mc = getattr(config, 'matching', config)
    entity_mismatch_penalty = getattr(mc, 'entity_mismatch_penalty', 0.25)  # Default penalty

    if entity_mismatch_penalty <= 0:
        return confidence, ""

    # Get voiceover text
    vo_text = vo_segment.text.lower() if vo_segment.text else ""

    # Get video path components
    video_file = video_segment.source_file.lower() if video_segment.source_file else ""
    source_keyword = getattr(video_segment, 'source_keyword', '')
    source_keyword = source_keyword.lower() if source_keyword else ""

    # Extract folder name from path
    from pathlib import Path
    video_path = Path(video_file) if video_file else None
    folder_name = video_path.parent.name.lower() if video_path else ""

    # Check if video path/folder contains any entity name
    video_entity = None
    for entity in known_entities:
        entity_lower = entity.lower()
        # Check if entity is in the video path/folder/keyword
        if (entity_lower in folder_name or
            entity_lower in video_file or
            entity_lower in source_keyword):
            video_entity = entity
            break

    if not video_entity:
        # Video is not entity-specific, no penalty
        return confidence, ""

    # Check if this entity is mentioned in the voiceover segment
    if video_entity.lower() in vo_text:
        # Entity IS mentioned, no penalty (let boost handle it)
        return confidence, ""

    # Entity video being used for non-entity voiceover segment - apply penalty
    penalized = max(0.0, confidence - entity_mismatch_penalty)
    reason = f"entity mismatch penalty: -{entity_mismatch_penalty:.2f} ({video_entity} not in VO)"

    logger.debug(f"Entity mismatch penalty applied: {confidence:.2f} -> {penalized:.2f} [{video_entity}]")

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
