"""
LLM-based reranking of embedding search candidates.

Extracted from tiered_matcher.py for single responsibility: LLM reranking.
Selects the best video segment from candidates using LLM semantic understanding.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import logging
from typing import TYPE_CHECKING, Dict, List, Tuple, Optional, Any

from .llm_providers import validate_llm_reasoning
from .scoring import compute_adaptive_context_weights
from .similarity_cache import get_video_context_cache, video_context_cache_key

if TYPE_CHECKING:
    from ..utils import SRTSegment
    from .llm_providers import LLMProvider

logger = logging.getLogger(__name__)


# US-134-003: Generic/low-signal tags to filter out
GENERIC_LOW_SIGNAL_TAGS = {
    'video', 'videos', 'youtube', 'yt', 'vlog', 'vlogs',
    'upload', 'uploaded', 'channel', 'subscribe', 'like',
    'comment', 'comments', 'share', 'sharing', 'music', 'song',
    'official', 'hd', 'hd', '4k', '1080p', '720p', 'new', 'new video',
    'official video', 'music video', 'lyric video', 'visual', 'audio',
    'clip', 'clips', 'footage', '素材', '视频', 'youtube video'
}


def filter_and_score_tags(
    tags: List[str],
    title: Optional[str] = None,
    description: Optional[str] = None,
    max_tags: int = 5
) -> List[str]:
    """Filter and score tags by relevance for context enrichment (US-134-003).

    Filters out generic/low-signal tags and prioritizes tags that appear in
    title or description for more specific context.

    Args:
        tags: Raw list of video tags
        title: Video title for relevance scoring
        description: Video description for relevance scoring
        max_tags: Maximum number of tags to return

    Returns:
        Filtered and scored list of tags (up to max_tags)
    """
    if not tags:
        return []

    # Convert to lowercase for comparison
    tags_lower = {tag.lower().strip(): tag for tag in tags if tag and tag.strip()}

    # Remove generic/low-signal tags
    filtered = [
        original for lower, original in tags_lower.items()
        if lower not in GENERIC_LOW_SIGNAL_TAGS
        and not any(generic in lower for generic in GENERIC_LOW_SIGNAL_TAGS)
    ]

    if not filtered:
        return []

    # Score tags by relevance (appears in title/description = higher score)
    def tag_relevance_score(tag: str) -> float:
        score = 0.0
        tag_lower = tag.lower()

        # Bonus if tag appears in title (high relevance)
        if title and tag_lower in title.lower():
            score += 2.0

        # Smaller bonus if tag appears in description
        if description and tag_lower in description.lower():
            score += 1.0

        # Prefer longer/more specific tags (likely more descriptive)
        if len(tag) > 5:
            score += 0.5

        # Prefer tags with spaces (multi-word = more specific)
        if ' ' in tag:
            score += 0.5

        return score

    # Sort by relevance score (descending)
    scored_tags = sorted(filtered, key=tag_relevance_score, reverse=True)

    return scored_tags[:max_tags]


# US-141-004: Topic categories for cross-signal validation
TOPIC_CATEGORIES = {
    'tutorial': {'tutorial', 'how to', 'guide', 'learn', 'course', 'lesson', 'tutoriales', 'tutoriel', 'anleitung'},
    'gaming': {'gaming', 'gameplay', 'gamer', 'playthrough', 'walkthrough', 'twitch', 'stream', 'esports', 'videogame', 'video game'},
    'music': {'music', 'song', 'album', 'artist', 'band', 'concert', 'lyrics', 'musica', 'musique'},
    'news': {'news', 'breaking', 'update', 'report', 'journalism', 'anchor'},
    'sports': {'sports', 'football', 'basketball', 'soccer', 'baseball', 'nfl', 'nba', 'mlb', 'game', 'match'},
    'cooking': {'cooking', 'recipe', 'food', 'kitchen', 'baking', 'cook', 'chef', 'dinner', 'meal'},
    'tech': {'tech', 'technology', 'review', 'unboxing', 'setup', 'computer', 'phone', 'laptop', 'gadget'},
    'entertainment': {'entertainment', 'movie', 'film', 'trailer', 'comedy', 'funny', 'sketch', 'show'},
    'vlog': {'vlog', 'daily', 'life', 'vlogger', 'vlogging', 'day in', 'routine'},
    'education': {'education', 'educational', 'school', 'university', 'college', 'lecture', 'documentary'},
}


def validate_context_consistency(
    title: str,
    description: str,
    tags: List[str],
    penalty_max: float = 0.05
) -> Tuple[float, float]:
    """Validate consistency between title, description, and tags (US-141-004).

    Detects when signals contradict each other (e.g., title says 'tutorial' but tags
    say 'gaming'). Returns a consistency score (0-1) and penalty to apply to
    confidence when signals are inconsistent.

    Args:
        title: Video title
        description: Video description
        tags: List of video tags
        penalty_max: Maximum penalty to apply when signals are inconsistent (default 0.05)

    Returns:
        Tuple of (consistency_score: float, penalty: float)
        - consistency_score: 1.0 = all signals consistent, 0.0 = contradictory
        - penalty: Amount to reduce confidence (0.0 to penalty_max)
    """
    if not title and not description and not tags:
        # No signals to validate - return full consistency
        return 1.0, 0.0

    # Extract topic keywords from each signal
    title_lower = (title or '').lower()
    desc_lower = (description or '').lower() if description else ''
    tags_lower = set((tag or '').lower() for tag in (tags or []) if tag)

    # Find detected topics in each signal
    def detect_topics(text: str) -> set:
        """Detect topic categories in text."""
        detected = set()
        for category, keywords in TOPIC_CATEGORIES.items():
            if any(kw in text for kw in keywords):
                detected.add(category)
        return detected

    title_topics = detect_topics(title_lower)
    desc_topics = detect_topics(desc_lower)
    tag_topics = detect_topics(' '.join(tags_lower))

    # Combine all topics found
    all_topics = title_topics | desc_topics | tag_topics

    if not all_topics:
        # No recognizable topics - can't detect contradiction
        return 1.0, 0.0

    # Calculate per-signal topic consistency
    # A signal is consistent if its topics overlap with the combined set
    signals = [
        ('title', title_topics),
        ('description', desc_topics),
        ('tags', tag_topics),
    ]

    consistency_scores = []
    for signal_name, signal_topics in signals:
        if signal_topics:
            # How many of this signal's topics are in the combined set?
            overlap = len(signal_topics & all_topics)
            signal_score = overlap / len(signal_topics) if signal_topics else 1.0
            consistency_scores.append(signal_score)

    # Also check cross-signal contradictions
    # If title and tags have different dominant topics, that's a contradiction
    contradiction_penalty = 0.0
    if title_topics and tag_topics:
        # Check if there's any overlap between title topics and tag topics
        title_tag_overlap = len(title_topics & tag_topics)
        if title_tag_overlap == 0 and len(title_topics) > 0 and len(tag_topics) > 0:
            # Title and tags have completely different topic categories
            contradiction_penalty = 0.3  # Moderate penalty for clear contradiction

    if title_topics and desc_topics:
        title_desc_overlap = len(title_topics & desc_topics)
        if title_desc_overlap == 0 and len(title_topics) > 0 and len(desc_topics) > 0:
            contradiction_penalty = max(contradiction_penalty, 0.2)

    # Calculate average consistency
    avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 1.0

    # Apply contradiction penalty to consistency score
    final_consistency = max(0.0, avg_consistency - contradiction_penalty)

    # Convert consistency to penalty (inverted: low consistency = high penalty)
    # Only apply penalty if consistency is below threshold
    if final_consistency < 0.8:
        # Scale penalty based on how low the consistency is
        penalty = penalty_max * (1.0 - final_consistency)
    else:
        penalty = 0.0

    return final_consistency, min(penalty, penalty_max)


@dataclass
class ContextPriorityWeightsConfig:
    """Context priority weights for video metadata signals (US-111-007).

    Controls how much weight the LLM should give to different context signals
    when evaluating video candidates. Higher values = higher priority.
    """
    title: float = 0.35  # Title is most important - primary topic indicator
    description: float = 0.30  # Description provides context
    tags: float = 0.20  # Tags are keywords but may be less reliable
    chapters: float = 0.15  # Chapters help with segment timing

    def __post_init__(self):
        # Normalize weights to sum to 1.0 if they don't
        total = self.title + self.description + self.tags + self.chapters
        if total > 0 and abs(total - 1.0) > 0.01:
            self.title = self.title / total
            self.description = self.description / total
            self.tags = self.tags / total
            self.chapters = self.chapters / total


@dataclass
class LLMRerankerConfig:
    """Configuration for LLM reranking."""
    ambiguous_threshold: float = 0.65  # Confidence below this triggers secondary LLM
    cache_llm_responses: bool = True  # Enable response caching

    # US-95-005: Include video metadata (title, description, tags, chapters) in context
    reranker_include_metadata: bool = True  # Pass full metadata to LLM for better context

    # US-111-007: Context priority weights for metadata signals
    context_priority_weights: Optional[ContextPriorityWeightsConfig] = None

    # US-134-002: Enable adaptive weighting based on available metadata
    adaptive_context_weights: bool = False

    # US-134-007: Transcript context for segment matching
    transcript_context_enabled: bool = True  # Include transcript snippets in LLM context
    transcript_context_chars: int = 200  # Max characters of transcript context to include

    # US-134-011: Context cache TTL for video context building
    context_cache_ttl_seconds: float = 3600.0  # TTL for video context cache

    # Confidence calibration based on candidate spread (US-63-008)
    close_spread_threshold: float = 0.05  # If top-2 spread < this, apply reduction
    clear_winner_threshold: float = 0.20  # If top-2 spread > this, apply boost
    close_spread_factor: float = 0.9  # Multiply confidence by this when close spread
    clear_winner_factor: float = 1.1  # Multiply confidence by this when clear winner

    # US-84-006: LLM reasoning quality penalty
    low_quality_reasoning_penalty: float = 0.05  # Penalty when reasoning is generic
    secondary_llm_decay: float = 0.9  # Multiplier when secondary fallback triggered

    # US-141-004: Cross-signal validation for context quality
    # Detects when title/description/tags contradict each other
    cross_signal_validation_enabled: bool = True  # Enable cross-signal consistency check
    consistency_penalty_max: float = 0.05  # Maximum penalty when signals are inconsistent

    def __post_init__(self):
        # US-111-007: Context priority weights nested config conversion
        if self.context_priority_weights is None:
            self.context_priority_weights = ContextPriorityWeightsConfig()
        elif isinstance(self.context_priority_weights, dict):
            self.context_priority_weights = ContextPriorityWeightsConfig(**self.context_priority_weights)


@dataclass
class RerankResult:
    """Result of LLM reranking."""
    selected_idx: int  # Index of selected candidate
    confidence: float  # Confidence score (0.0-1.0)
    reasoning: str  # Explanation for the selection
    used_secondary: bool = False  # Whether secondary provider was used
    llm_reasoning_quality: int = 1  # 0=low quality reasoning, 1=normal


class LLMReranker:
    """
    LLM-based reranking for video segment selection.

    Uses LLM semantic understanding to select the best video segment
    from embedding-filtered candidates. Supports primary and secondary
    providers with automatic fallback for ambiguous matches.
    """

    def __init__(
        self,
        config: LLMRerankerConfig,
        cache: Optional[Any] = None
    ):
        """Initialize LLM reranker with config."""
        self.config = config
        self.cache = cache

        # US-134-011: Initialize video context cache
        # Get TTL from config (passed from matching_config)
        ttl_seconds = getattr(config, 'context_cache_ttl_seconds', 3600.0)
        self._video_context_cache = get_video_context_cache(ttl_seconds=ttl_seconds)

    @classmethod
    def from_matching_config(
        cls,
        matching_config: Any,
        cache: Optional[Any] = None
    ) -> 'LLMReranker':
        """Create LLMReranker from matching config section."""
        # US-111-007: Read context_priority_weights from matching config
        priority_weights_dict = getattr(matching_config, 'context_priority_weights', None)
        context_priority_weights = None
        if priority_weights_dict:
            context_priority_weights = ContextPriorityWeightsConfig(**priority_weights_dict)

        config = LLMRerankerConfig(
            ambiguous_threshold=getattr(matching_config, 'ambiguous_threshold', 0.65),
            cache_llm_responses=getattr(matching_config, 'cache_llm_responses', True),
            reranker_include_metadata=getattr(matching_config, 'reranker_include_metadata', True),  # US-95-005
            context_priority_weights=context_priority_weights,  # US-111-007
            adaptive_context_weights=getattr(matching_config, 'adaptive_context_weights', False),  # US-134-002
            transcript_context_enabled=getattr(matching_config, 'transcript_context_enabled', True),  # US-134-007
            transcript_context_chars=getattr(matching_config, 'transcript_context_chars', 200),  # US-134-007
            context_cache_ttl_seconds=getattr(matching_config, 'context_cache_ttl_seconds', 3600.0),  # US-134-011
            close_spread_threshold=getattr(matching_config, 'llm_reranker_close_spread_threshold', 0.05),
            clear_winner_threshold=getattr(matching_config, 'llm_reranker_clear_winner_threshold', 0.20),
            close_spread_factor=getattr(matching_config, 'llm_reranker_close_spread_factor', 0.9),
            clear_winner_factor=getattr(matching_config, 'llm_reranker_clear_winner_factor', 1.1),
            cross_signal_validation_enabled=getattr(matching_config, 'cross_signal_validation_enabled', True),  # US-141-004
            consistency_penalty_max=getattr(matching_config, 'consistency_penalty_max', 0.05),  # US-141-004
        )
        return cls(config, cache)

    def rerank(
        self,
        voiceover_text: str,
        candidates: List[Tuple['SRTSegment', float]],
        primary_provider: Optional['LLMProvider'] = None,
        secondary_provider: Optional['LLMProvider'] = None,
        context: Optional[str] = None,
        negative_rules: Optional[List[str]] = None,
        video_metadata: Optional[Dict[str, Dict[str, str]]] = None,
        transcript_data: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> RerankResult:
        """
        Rerank candidates using LLM semantic understanding.

        Args:
            voiceover_text: Text of the voiceover segment
            candidates: List of (video_segment, similarity) tuples
            context: Optional context string
            negative_rules: Optional list of things to avoid
            video_metadata: Optional dict mapping source_file (video ID) to
                {"title": str, "description": str} for context enrichment
            transcript_data: Optional dict mapping source_file to list of transcript
                segments with 'text', 'start', 'end' for transcript context (US-134-007)

        Returns:
            RerankResult with selected index, confidence, and reasoning
        """
        if not candidates:
            return RerankResult(selected_idx=0, confidence=0.0, reasoning="No candidates provided")

        # Enrich candidates with video context for LLM prompt (US-70-007)
        # US-134-007: Also enrich with transcript context
        enriched_candidates = self._enrich_candidates_with_context(
            candidates[:5], video_metadata, transcript_data
        )

        # Check cache first (uses enriched text for cache key)
        cache_key = self._get_cache_key(voiceover_text, enriched_candidates)
        cached = self._get_cached_response(cache_key)
        if cached:
            logger.info(f"LLM reranker: cache hit, using cached result")
            return RerankResult(
                selected_idx=cached[0],
                confidence=cached[1],
                reasoning=f"(cached) {cached[2]}"
            )

        # No LLM provider - return embedding fallback
        if not primary_provider:
            embedding_sim = candidates[0][1] if candidates else 0.5
            logger.info(f"LLM reranker: no provider, using embedding fallback (sim={embedding_sim:.2f})")
            return RerankResult(
                selected_idx=0,
                confidence=0.60,
                reasoning=f"Embedding similarity only (sim={embedding_sim:.2f})"
            )

        # Call primary provider with enriched candidates
        # Extract video IDs for logging
        candidate_vids = [getattr(cand[0], 'source_file', 'unknown')[:15] for cand in candidates[:3]]

        # US-162-007: Debug logging for LLM request sizes
        voiceover_len = len(voiceover_text)
        candidates_text_len = sum(len(cand[0].text) for cand in candidates[:5])
        context_len = len(context) if context else 0
        logger.info(f"[LLM_RERANK] calling primary with {len(candidates)} candidates, video_ids={candidate_vids}")
        logger.debug(f"[LLM_RERANK_DEBUG] Request size - voiceover: {voiceover_len} chars, candidates: {candidates_text_len} chars (top 5), context: {context_len} chars")

        try:
            results = primary_provider.match_batch(
                [(voiceover_text, enriched_candidates)],
                context=context,
                negative_rules=negative_rules
            )
            selected_idx, confidence, reasoning, _cot = results[0]

            # US-162-007: Debug logging for LLM response sizes
            reasoning_len = len(reasoning) if reasoning else 0
            logger.debug(f"[LLM_RERANK_DEBUG] Response size - selected_idx: {selected_idx}, confidence: {confidence:.3f}, reasoning: {reasoning_len} chars")

            # Check for ambiguous match - use secondary provider
            used_secondary = False
            if confidence < self.config.ambiguous_threshold and secondary_provider:
                logger.debug(f"Ambiguous match ({confidence:.2f}), using secondary LLM")
                secondary_results = secondary_provider.match_batch(
                    [(voiceover_text, enriched_candidates)],
                    context=context,
                    negative_rules=negative_rules
                )
                sec_idx, sec_conf, sec_reason, _ = secondary_results[0]
                sec_reasoning_len = len(sec_reason) if sec_reason else 0
                logger.debug(f"[LLM_RERANK_DEBUG] Secondary response - confidence: {sec_conf:.3f}, reasoning: {sec_reasoning_len} chars")

                if sec_conf > confidence:
                    selected_idx = sec_idx
                    confidence = sec_conf
                    reasoning = f"(secondary) {sec_reason}"
                    used_secondary = True
                    logger.info(f"[LLM_RERANK] used secondary provider, confidence={confidence:.3f}")

            # Apply confidence calibration based on candidate spread (US-63-008)
            confidence, spread_adjustment = self._apply_spread_calibration(
                confidence, candidates
            )
            if spread_adjustment != 0.0:
                reasoning = f"{reasoning} [spread_adj={spread_adjustment:+.2f}]"

            # US-84-006: Validate LLM reasoning quality
            reasoning_validation = validate_llm_reasoning(
                reasoning=reasoning,
                voiceover_text=voiceover_text,
            )
            llm_reasoning_quality = 1 if reasoning_validation.is_valid else 0

            if not reasoning_validation.is_valid:
                penalty = self.config.low_quality_reasoning_penalty
                confidence = max(0.0, confidence - penalty)
                logger.debug(
                    f"US-84-006: Low-quality LLM reasoning penalty -{penalty} applied "
                    f"(refs={reasoning_validation.specific_references})"
                )

            # US-84-006: Apply secondary LLM decay multiplier
            if used_secondary:
                decay = self.config.secondary_llm_decay
                confidence = confidence * decay
                logger.debug(
                    f"US-84-006: Secondary LLM decay {decay}x applied"
                )

            # Cache the result
            self._cache_response(cache_key, selected_idx, confidence, reasoning)

            # Log final reranking summary (US-159-008)
            selected_vid = candidates[selected_idx][0].source_file if selected_idx < len(candidates) else 'unknown'
            logger.info(
                f"[LLM_RERANK] completed - selected_idx={selected_idx}, video_id={selected_vid}, confidence={confidence:.3f}, "
                f"used_secondary={used_secondary}, quality={llm_reasoning_quality}"
            )

            return RerankResult(
                selected_idx=selected_idx,
                confidence=confidence,
                reasoning=reasoning,
                used_secondary=used_secondary,
                llm_reasoning_quality=llm_reasoning_quality,
            )

        except Exception as e:
            logger.warning(f"LLM reranking failed: {e}, using embedding fallback")
            embedding_sim = candidates[0][1] if candidates else 0.5
            return RerankResult(
                selected_idx=0,
                confidence=0.60,
                reasoning=f"LLM fallback (emb_sim={embedding_sim:.2f})"
            )

    @staticmethod
    def _build_video_context(
        title: str,
        description: str,
        tags: Optional[List[str]] = None,
        chapters: Optional[List[Dict[str, Any]]] = None,
        priority_weights: Optional[ContextPriorityWeightsConfig] = None,
        adaptive: bool = False
    ) -> str:
        """Build video context string from title, description, tags, and chapters (US-95-005, US-111-007, US-134-002).

        Format: 'Video context: [Priority Signals] Title: {...} Description: {...} Tags: {...} Chapters: {...}'
        Returns empty string if no title/description/tags/chapters available.

        Args:
            title: Video title
            description: Video description
            tags: List of video tags
            chapters: List of chapter dicts with 'title' key
            priority_weights: Context priority weights for formatting emphasis
            adaptive: Whether to use adaptive weights based on available metadata (US-134-002)
        """
        if not title and not description and not tags and not chapters:
            return ""

        # Default priority weights if not provided
        if priority_weights is None:
            priority_weights = ContextPriorityWeightsConfig()

        # Track which signals are available
        available_signals = []

        # US-134-002: Check which signals are actually available
        has_title = bool(title and title.strip())
        has_description = bool(description and description.strip())
        has_tags = bool(tags and len(tags) > 0)
        has_chapters = bool(chapters and len(chapters) > 0)

        # US-134-002: Compute adaptive weights if enabled
        if adaptive:
            base_weights = (
                priority_weights.title,
                priority_weights.description,
                priority_weights.tags,
                priority_weights.chapters
            )
            title_w, desc_w, tags_w, chapters_w = compute_adaptive_context_weights(
                base_weights,
                has_title=has_title,
                has_description=has_description,
                has_tags=has_tags,
                has_chapters=has_chapters
            )
            # Create a new weights object with adaptive values
            priority_weights = ContextPriorityWeightsConfig(
                title=title_w,
                description=desc_w,
                tags=tags_w,
                chapters=chapters_w
            )

        # Title is highest priority - include first
        if has_title:
            available_signals.append(f"Title: {title}")

        # Description provides context - include second
        if has_description:
            # Extract first sentence (up to 100 chars)
            first_sentence = description.split('.')[0].strip()
            if len(first_sentence) > 100:
                first_sentence = first_sentence[:97] + "..."
            if first_sentence:
                available_signals.append(f"Description: {first_sentence}")

        # Tags are keywords but may be less reliable
        # US-134-003: Use filtered and relevance-scored tags
        if has_tags:
            # Filter out generic tags and score by relevance to title/description
            filtered_tags = filter_and_score_tags(
                tags,
                title=title,
                description=description,
                max_tags=5
            )
            if filtered_tags:
                tag_str = ", ".join(filtered_tags)
                available_signals.append(f"Tags: {tag_str}")

        # Chapters help with segment timing - lowest priority
        if has_chapters:
            # Extract chapter titles (skip timestamps)
            chapter_titles = []
            for ch in chapters[:3]:  # Take up to 3 chapters
                if isinstance(ch, dict):
                    ch_title = ch.get('title', '')
                else:
                    ch_title = str(ch)
                if ch_title and ch_title != 'Unknown':
                    chapter_titles.append(ch_title)
            if chapter_titles:
                available_signals.append(f"Chapters: {', '.join(chapter_titles)}")

        if not available_signals:
            return ""

        # US-111-007: Format with priority weights indicator and signal breakdown
        # This helps the LLM understand which signals to prioritize
        weights_str = f"[weights: title={priority_weights.title:.0%}, desc={priority_weights.description:.0%}, tags={priority_weights.tags:.0%}, chapters={priority_weights.chapters:.0%}]"

        return f"Video context {weights_str}: " + " | ".join(available_signals)

    def _enrich_candidates_with_context(
        self,
        candidates: List[Tuple['SRTSegment', float]],
        video_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        transcript_data: Optional[Dict[str, List[Dict[str, Any]]]] = None
    ) -> List[Tuple['SRTSegment', float]]:
        """Enrich candidate segments with video title/description/tags/chapters context (US-70-007, US-95-005).

        Also enriches with transcript context when available (US-134-007).

        Creates shallow copies of SRTSegments with enriched text that includes
        video context prefix when metadata is available. Falls back to original
        text when no metadata exists for a candidate.

        Args:
            candidates: List of (segment, similarity) tuples
            video_metadata: Dict mapping source_file to {title, description, tags, chapters}
            transcript_data: Dict mapping source_file to list of transcript segments
                with 'text', 'start', 'end' keys
        """
        # US-95-005: Check config option to enable/disable metadata enrichment
        if not video_metadata and not transcript_data:
            return candidates

        # US-134-007: Check if we have any enrichment to do
        has_metadata = video_metadata and self.config.reranker_include_metadata
        has_transcript = transcript_data and self.config.transcript_context_enabled

        if not has_metadata and not has_transcript:
            return candidates

        enriched = []
        # US-134-011: Track built contexts per video to avoid rebuilding
        video_context_cache: Dict[str, str] = {}

        for seg, sim in candidates:
            context_parts = []

            # US-95-005: Add video metadata context
            if has_metadata:
                video_id = seg.source_file

                # US-134-011: Check cache first
                if video_id in video_context_cache:
                    cached_context = video_context_cache[video_id]
                    if cached_context:
                        context_parts.append(cached_context)
                else:
                    # Build and cache the context
                    meta = video_metadata.get(video_id, {})
                    title = meta.get('title', '')
                    description = meta.get('description', '')
                    tags = meta.get('tags', [])
                    chapters = meta.get('chapters', '')

                    # Generate cache key using video_id
                    cache_key = video_context_cache_key(video_id)

                    # Try to get from cache
                    cached = self._video_context_cache.get(cache_key)
                    if cached is not None:
                        video_context_cache[video_id] = cached
                        if cached:
                            context_parts.append(cached)
                    else:
                        # Build the context (expensive operation)
                        video_context = self._build_video_context(
                            title, description, tags, chapters,
                            priority_weights=self.config.context_priority_weights,
                            adaptive=self.config.adaptive_context_weights
                        )

                        # Store in both local and global cache
                        video_context_cache[video_id] = video_context
                        if video_context:
                            self._video_context_cache.put(cache_key, video_context)
                            context_parts.append(video_context)

                # US-134-007: Extract transcript segments from video_metadata
                # Note: Transcript context is per-segment (not cached) since it depends on segment timestamp
                video_transcript = None
                if has_transcript and transcript_data:
                    video_transcript = transcript_data.get(video_id, [])
                elif not video_transcript:
                    # Also check in video_metadata directly (US-134-007)
                    meta = video_metadata.get(video_id, {})
                    video_transcript = meta.get('transcript_segments', [])

                if video_transcript:
                    transcript_context = self._extract_transcript_context(seg, video_transcript)
                    if transcript_context:
                        context_parts.append(f"Transcript: {transcript_context}")

            if context_parts:
                enriched_seg = copy.copy(seg)
                enriched_seg.text = f"[{' | '.join(context_parts)}] {seg.text}"
                enriched.append((enriched_seg, sim))
            else:
                enriched.append((seg, sim))

        return enriched

    def _extract_transcript_context(
        self,
        segment: 'SRTSegment',
        video_transcript: List[Dict[str, Any]]
    ) -> str:
        """Extract transcript context around a segment's timestamp (US-134-007).

        Finds transcript segments within a time window around the given segment
        and extracts key phrases. Limits output to transcript_context_chars.

        Args:
            segment: The video segment to find context for
            video_transcript: List of transcript segments with 'text', 'start', 'end'

        Returns:
            Transcript context string (up to transcript_context_chars) or empty string
        """
        if not video_transcript or not self.config.transcript_context_enabled:
            return ""

        seg_start = segment.start_time
        seg_end = segment.end_time

        # Find segments within the time window around this segment
        # Use a window of +/- 30 seconds around the segment
        window_start = seg_start - 30
        window_end = seg_end + 30

        context_parts = []
        for ts in video_transcript:
            ts_start = ts.get('start', 0)
            ts_end = ts.get('end', 0)
            ts_text = ts.get('text', '')

            # Skip segments outside the window
            if ts_end < window_start or ts_start > window_end:
                continue

            # Skip the segment itself (we want surrounding context, not the segment)
            if ts_start >= seg_start and ts_end <= seg_end:
                continue

            if ts_text:
                context_parts.append(ts_text)

        if not context_parts:
            return ""

        # Join and truncate to max chars
        context_str = " ".join(context_parts)
        max_chars = self.config.transcript_context_chars

        if len(context_str) > max_chars:
            context_str = context_str[:max_chars] + "..."

        return context_str

    def _get_cache_key(self, voiceover_text: str, candidates: List[Tuple['SRTSegment', float]]) -> str:
        """Generate cache key for LLM response."""
        content = voiceover_text + "|" + "|".join(c[0].text for c in candidates[:5])
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _get_cached_response(self, cache_key: str) -> Optional[Tuple[int, float, str]]:
        """Get cached LLM response."""
        if not self.config.cache_llm_responses or not self.cache:
            return None
        cached = self.cache.get_llm_response(cache_key)
        if cached:
            return (cached['selected'], cached['confidence'], cached['reasoning'])
        return None

    def _cache_response(self, cache_key: str, selected: int, confidence: float, reasoning: str) -> None:
        """Cache LLM response."""
        if self.config.cache_llm_responses and self.cache:
            self.cache.save_llm_response(cache_key, {
                'selected': selected,
                'confidence': confidence,
                'reasoning': reasoning
            })

    def get_context_cache_stats(self) -> Dict[str, Any]:
        """Get video context cache statistics (US-134-011)."""
        return self._video_context_cache.get_stats()

    def _apply_spread_calibration(
        self,
        confidence: float,
        candidates: List[Tuple['SRTSegment', float]]
    ) -> Tuple[float, float]:
        """
        Apply confidence calibration based on candidate spread (US-63-008).

        If top-2 candidates are very close (spread < close_spread_threshold),
        reduce confidence to reflect ambiguity.

        If top-2 candidates are far apart (spread > clear_winner_threshold),
        boost confidence to reflect certainty (capped at 1.0).

        Args:
            confidence: Raw confidence from LLM
            candidates: List of (segment, similarity) tuples

        Returns:
            Tuple of (calibrated_confidence, adjustment_delta)
        """
        if len(candidates) < 2:
            logger.debug("US-63-008 spread calibration: skipped (< 2 candidates)")
            return confidence, 0.0

        # Calculate spread between top-2 candidate similarities
        top_sim = candidates[0][1]
        second_sim = candidates[1][1]
        spread = top_sim - second_sim

        original_confidence = confidence
        adjustment = 0.0

        if spread < self.config.close_spread_threshold:
            # Very close candidates - reduce confidence (ambiguity)
            confidence = confidence * self.config.close_spread_factor
            adjustment = confidence - original_confidence
            logger.debug(
                f"US-63-008 spread calibration: close spread ({spread:.3f} < {self.config.close_spread_threshold}), "
                f"confidence {original_confidence:.3f} -> {confidence:.3f} (factor={self.config.close_spread_factor})"
            )
        elif spread > self.config.clear_winner_threshold:
            # Clear winner - boost confidence (capped at 1.0)
            confidence = min(1.0, confidence * self.config.clear_winner_factor)
            adjustment = confidence - original_confidence
            logger.debug(
                f"US-63-008 spread calibration: clear winner ({spread:.3f} > {self.config.clear_winner_threshold}), "
                f"confidence {original_confidence:.3f} -> {confidence:.3f} (factor={self.config.clear_winner_factor}, capped=1.0)"
            )
        else:
            logger.debug(
                f"US-63-008 spread calibration: neutral spread ({spread:.3f}), no adjustment"
            )

        return confidence, adjustment
