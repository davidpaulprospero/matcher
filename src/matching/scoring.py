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

from typing import Any, Tuple, List, Optional
import logging
import statistics

from ..utils import SRTSegment
from ..topic_extraction import compute_topic_penalty

logger = logging.getLogger(__name__)


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
    if candidates and len(candidates) >= 2:
        top_scores = [sim for _, sim in candidates[:5]]
        try:
            variance = statistics.stdev(top_scores) if len(top_scores) >= 2 else 0.0
        except statistics.StatisticsError:
            variance = 0.0

        if variance < 0.05:
            adjustment -= 0.05
            reasons.append(f"low_var({variance:.3f}):-0.05")

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


def apply_caption_quality_adjustment(
    confidence: float,
    video_segment: SRTSegment,
    config
) -> Tuple[float, str]:
    """
    Apply confidence adjustment based on caption quality (US-007, US-006).

    Supports two modes:
    1. Multiplicative weights (US-006): When caption_quality_weights dict is set,
       applies: adjusted = raw_confidence * weight
       Example: {high: 1.0, medium: 0.9, low: 0.75}

    2. Additive boost/penalty (US-007): When caption_quality_weights is None,
       uses high_boost and low_penalty for additive adjustments.

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
        # Multiplicative weights mode (US-006)
        # Default weights if not specified: high=1.0, medium=0.9, low=0.75
        default_weights = {'high': 1.0, 'medium': 0.9, 'low': 0.75}
        weight = quality_weights.get(caption_quality, default_weights.get(caption_quality, 1.0))

        if weight != 1.0:
            adjusted = max(0.0, min(1.0, confidence * weight))
            reason = f"caption quality {caption_quality}: x{weight:.2f}"
            # US-006: Specific log format requested
            logger.info(f"Confidence adjusted {confidence:.2f} -> {adjusted:.2f} ({caption_quality} quality caption)")
            return adjusted, reason

        return confidence, ""

    # Legacy additive mode (US-007) - when caption_quality_weights is None
    high_boost = getattr(mc, 'caption_quality_high_boost', 0.05)
    low_penalty = getattr(mc, 'caption_quality_low_penalty', 0.1)

    if caption_quality == 'high' and high_boost > 0:
        adjusted = min(1.0, confidence + high_boost)
        reason = f"caption quality high: +{high_boost:.2f}"
        logger.debug(f"Caption quality boost applied: {confidence:.2f} -> {adjusted:.2f}")
        return adjusted, reason

    elif caption_quality == 'low' and low_penalty > 0:
        adjusted = max(0.0, confidence - low_penalty)
        reason = f"caption quality low: -{low_penalty:.2f}"
        logger.debug(f"Caption quality penalty applied: {confidence:.2f} -> {adjusted:.2f}")
        return adjusted, reason

    # Medium quality or unknown - no adjustment
    return confidence, ""


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
    video_segment: SRTSegment
) -> Tuple[float, str, List[str]]:
    """
    Apply confidence boost when video contains same named entities as voiceover.

    Named entities (people, places, organizations) are strong signals for
    video-voiceover matching. A video mentioning the same person or place
    as the voiceover is highly relevant.

    Graduated boost values:
    - 1 matching entity: +0.05
    - 2 matching entities: +0.08
    - 3+ matching entities: +0.12

    Args:
        confidence: Original confidence score
        vo_segment: Voiceover segment (may have entities from analysis)
        video_segment: Video segment being considered

    Returns:
        Tuple of (boosted_confidence, boost_reason, matched_entities)
    """
    # Extract entity texts from voiceover segment
    vo_entities = _extract_entity_texts(vo_segment)
    if not vo_entities:
        return confidence, "", []

    # Extract entity texts from video segment
    video_entities = _extract_entity_texts(video_segment)
    if not video_entities:
        return confidence, "", []

    # Find matching entities (case-insensitive)
    vo_lower = {e.lower() for e in vo_entities}
    video_lower = {e.lower() for e in video_entities}
    matching = vo_lower & video_lower

    if not matching:
        return confidence, "", []

    # Graduated boost based on match count
    match_count = len(matching)
    if match_count >= 3:
        boost = 0.12
    elif match_count == 2:
        boost = 0.08
    else:
        boost = 0.05

    # Apply boost (cap at 1.0)
    boosted = min(1.0, confidence + boost)

    # Get original-case matched entity names for return
    matched_entities = [e for e in vo_entities if e.lower() in matching]

    reason = f"entity match: +{boost:.2f} ({match_count} entities: {', '.join(matched_entities[:3])})"

    logger.debug(f"Entity match boost applied: {confidence:.2f} -> {boosted:.2f} ({matched_entities})")

    return boosted, reason, matched_entities


def _extract_entity_texts(segment: SRTSegment) -> List[str]:
    """
    Extract entity text values from a segment.

    Entities are stored as dicts with 'text', 'type', and 'context' keys.
    Also checks keywords list for entity-like entries.

    Args:
        segment: SRTSegment to extract entities from

    Returns:
        List of entity text values (names)
    """
    entities = []

    # Get entities from the entities field
    segment_entities = getattr(segment, 'entities', []) or []
    for entity in segment_entities:
        if isinstance(entity, dict):
            text = entity.get('text', '')
            if text and len(text) >= 2:  # Skip very short entities
                entities.append(text)
        elif isinstance(entity, str):
            if entity and len(entity) >= 2:
                entities.append(entity)

    # Also check keywords for entity-like entries (proper nouns, capitalized words)
    keywords = getattr(segment, 'keywords', []) or []
    for kw in keywords:
        if kw and len(kw) >= 2:
            # Check if it looks like a proper noun (capitalized, multi-word)
            if _looks_like_entity(kw):
                entities.append(kw)

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

    # Factor 1: Word count scoring (0.0 - 0.4)
    # More words generally means better quality transcription
    words = text.split()
    word_count = len(words)

    if word_count >= TRANSCRIPT_MIN_WORD_COUNT_GOOD:
        word_score = 0.4
    elif word_count >= TRANSCRIPT_MIN_WORD_COUNT_MEDIUM:
        # Linear interpolation between 20 and 50 words
        word_score = 0.2 + 0.2 * ((word_count - TRANSCRIPT_MIN_WORD_COUNT_MEDIUM) /
                                   (TRANSCRIPT_MIN_WORD_COUNT_GOOD - TRANSCRIPT_MIN_WORD_COUNT_MEDIUM))
    else:
        # Linear interpolation between 0 and 20 words
        word_score = 0.2 * (word_count / TRANSCRIPT_MIN_WORD_COUNT_MEDIUM) if word_count > 0 else 0.0

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

    # Determine quality tier
    if quality_score >= TRANSCRIPT_QUALITY_HIGH_THRESHOLD:
        quality_tier = "high"
    elif quality_score >= TRANSCRIPT_QUALITY_MEDIUM_THRESHOLD:
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


def compute_multimodal_score(
    embedding_similarity: float,
    keyword_overlap_score: float,
    entity_match_score: float,
    visual_description_score: float,
    weights: dict = None,
    multimodal_enabled: bool = True
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

    # Use default weights if not provided
    w = weights if weights else DEFAULT_MULTIMODAL_WEIGHTS

    # Ensure weights sum to 1.0 (normalize if needed)
    weight_sum = sum(w.values())
    if abs(weight_sum - 1.0) > 0.01:
        logger.warning(f"Multimodal weights sum to {weight_sum:.3f}, normalizing to 1.0")
        w = {k: v / weight_sum for k, v in w.items()}

    # Clamp input scores to [0, 1] range
    emb_clamped = max(0.0, min(1.0, embedding_similarity))
    kw_clamped = max(0.0, min(1.0, keyword_overlap_score))
    ent_clamped = max(0.0, min(1.0, entity_match_score))
    vis_clamped = max(0.0, min(1.0, visual_description_score))

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

    logger.debug(
        f"Multimodal score: emb={emb_clamped:.3f}, kw={kw_clamped:.3f}, "
        f"ent={ent_clamped:.3f}, vis={vis_clamped:.3f} -> {multimodal_score:.3f}"
    )

    return multimodal_score, reason, component_scores


def calculate_keyword_overlap_score(
    vo_keywords: List[str],
    video_keywords: List[str]
) -> Tuple[float, List[str]]:
    """
    Calculate normalized keyword overlap score between voiceover and video.

    Args:
        vo_keywords: Keywords from voiceover segment
        video_keywords: Keywords from video segment

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
    # 1 match = 0.3, 2 matches = 0.5, 3 matches = 0.7, 4+ matches = 0.85-1.0
    match_count = len(matched)
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
    semantic_coherence_enabled: bool = True
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

    # Apply adjustments based on similarity thresholds
    if similarity > SEMANTIC_COHERENCE_SMOOTH_THRESHOLD:
        adjustment = SEMANTIC_COHERENCE_SMOOTH_BOOST
        reason = f"smooth_topic_flow(sim={similarity:.3f}):+{SEMANTIC_COHERENCE_SMOOTH_BOOST}"
    elif similarity < SEMANTIC_COHERENCE_ABRUPT_THRESHOLD:
        adjustment = -SEMANTIC_COHERENCE_ABRUPT_PENALTY
        reason = f"abrupt_topic_flow(sim={similarity:.3f}):-{SEMANTIC_COHERENCE_ABRUPT_PENALTY}"
    else:
        adjustment = 0.0
        reason = f"neutral_topic_flow(sim={similarity:.3f})"

    logger.debug(
        f"Semantic coherence: similarity={similarity:.3f}, adjustment={adjustment:+.3f} ({reason})"
    )

    return adjustment, reason


# Pool normalization constants
POOL_NORMALIZATION_REFERENCE_SIZE = 50  # Reference pool size for normalization
POOL_NORMALIZATION_MIN_FACTOR = 0.8  # Minimum normalization factor (caps boost)
POOL_NORMALIZATION_MAX_FACTOR = 1.2  # Maximum normalization factor (caps reduction)
POOL_SMALL_THRESHOLD = 10  # Pool considered "small" below this
POOL_LARGE_THRESHOLD = 100  # Pool considered "large" above this
POOL_TIGHT_MARGIN_THRESHOLD = 0.05  # Top-2 score difference threshold for "tight margin"


def normalize_confidence_by_pool(
    confidence: float,
    pool_size: int,
    candidates: Optional[List[Tuple[SRTSegment, float]]] = None,
    pool_normalization_enabled: bool = True
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

    reasons = []
    adjustment = 0.0

    # Calculate base normalization factor: sqrt(pool_size / 50)
    # Small pools: factor < 1.0 (confidence preserved/boosted)
    # Large pools: factor > 1.0 (confidence reduced)
    raw_factor = (pool_size / POOL_NORMALIZATION_REFERENCE_SIZE) ** 0.5

    # Clamp factor to [0.8, 1.2] range
    clamped_factor = max(POOL_NORMALIZATION_MIN_FACTOR,
                         min(POOL_NORMALIZATION_MAX_FACTOR, raw_factor))

    # Determine margin characteristics if candidates provided
    top_margin = None
    if candidates and len(candidates) >= 2:
        top_score = candidates[0][1]
        second_score = candidates[1][1]
        top_margin = top_score - second_score

    # Apply pool-specific adjustments
    if pool_size < POOL_SMALL_THRESHOLD:
        # Small pool: boost confidence if there's a clear winner
        if top_margin is not None and top_margin >= 0.1:
            # Clear winner in small pool - this is a strong signal
            adjustment = 0.05
            reasons.append(f"small_pool({pool_size})+clear_winner({top_margin:.2f}):+0.05")
        else:
            reasons.append(f"small_pool({pool_size}):factor={clamped_factor:.2f}")

    elif pool_size > POOL_LARGE_THRESHOLD:
        # Large pool: reduce confidence if margins are tight
        if top_margin is not None and top_margin < POOL_TIGHT_MARGIN_THRESHOLD:
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

    logger.debug(
        f"Pool normalization: pool={pool_size}, factor={clamped_factor:.2f}, "
        f"inverse={inverse_factor:.2f}, adj={adjustment:+.2f}, "
        f"{confidence:.3f} -> {normalized:.3f} ({reason})"
    )

    return normalized, reason
