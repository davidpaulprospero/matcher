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

from typing import Tuple, List, Optional
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
