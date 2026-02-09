"""Matching configuration: Matching engine, location matching, negative matching.

Extracted from monolithic config.py during refactoring (Jan 7, 2026).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    'LocationMatchingConfig',
    'NegativeMatchingConfig',
    'ChapterDetectionConfig',
    'MatchingScoringConfig',
    'ContextEnrichmentConfig',
    'ChapterGroupingConfig',
    'TieredCaptionPenalties',
    'MatchingConfig',
]


@dataclass
class LocationMatchingConfig:
    """Location-aware matching settings for geographic content.

    When chapters focus on specific locations (cities, countries, landmarks),
    this enables geographic filtering and scoring to ensure videos match
    the correct geographic context.

    Features:
    - Hard filtering by country (prevents Paris, TX matching Paris, FR content)
    - Soft penalty fallback when filter too strict
    - Hierarchy bonus (France video can match Paris chapter)
    - GeoNames API integration for geocoding
    """
    enabled: bool = True

    # GeoNames API (free account at geonames.org)
    geonames_username: str = ""  # Required for API calls

    # Filtering level: how strict the hard filter should be
    # Options: "city" (strictest), "state", "country", "continent" (loosest)
    hard_filter_level: str = "city"

    # Scoring adjustments
    geographic_penalty: float = 0.4     # Penalty for wrong location (soft mode fallback)
    hierarchy_bonus: float = 0.15       # Bonus for parent/child match (France for Paris)
    landmark_bonus: float = 0.2         # Bonus when landmark detected in video

    # Disambiguation
    use_llm_disambiguation: bool = True  # Use LLM to resolve ambiguous locations

    # Caching
    cache_dir: str = ".cache/locations"

    def __post_init__(self):
        # Read from environment variable if not set in config
        if not self.geonames_username:
            self.geonames_username = os.getenv("GEONAMES_USERNAME", "")


@dataclass
class NegativeMatchingConfig:
    """Negative matching rules - what NOT to match"""
    enabled: bool = True
    rules: List[str] = field(default_factory=lambda: [
        "Don't match talking head shots to action narration",
        "Don't match static images to dynamic narration",
        "Avoid matching unrelated b-roll to specific statements"
    ])


@dataclass
class ChapterDetectionConfig:
    """Enhanced chapter detection settings.

    Multi-pass chapter detection with:
    - Pass 1: Initial topic/location detection
    - Pass 2: Boundary refinement using embeddings
    - Pass 3: Cross-validation with LLM
    - Pass 4: Gap/overlap resolution
    """
    enabled: bool = True  # Enable enhanced multi-pass detection

    # Pass controls
    use_validation_pass: bool = True  # Pass 3: Cross-validate chapters
    use_boundary_refinement: bool = True  # Pass 2: Refine with embeddings

    # Strategy selection
    default_strategy: str = 'topic'  # 'topic' or 'location'
    auto_detect_content_type: bool = True  # Auto-switch strategy based on content

    # Chunking for long transcripts
    max_chunk_chars: int = 6000  # Max chars per LLM call
    chunk_overlap_segments: int = 5  # Segments to overlap between chunks

    # Chapter constraints
    min_chapter_confidence: float = 0.5  # Filter low-confidence chapters
    min_chapter_segments: int = 3  # Minimum segments per chapter
    max_chapters: int = 20  # Maximum chapters to detect


@dataclass
class MatchingScoringConfig:
    """Scoring thresholds and weights for match confidence calculations.

    Extracted from hardcoded constants in src/matching/scoring.py (US-53-002).
    All fields have defaults matching the original hardcoded values.
    """

    # Confidence floor and warning (applied in MatchScoring.apply_all_adjustments)
    confidence_floor: float = 0.05  # Minimum confidence after all penalties
    low_confidence_warning_threshold: float = 0.15  # Warn when penalized below this

    # Entity match boosts (graduated by match count)
    entity_match_boosts: Dict[str, float] = field(default_factory=lambda: {
        '1': 0.05,   # 1 entity match
        '2': 0.08,   # 2 entity matches
        '3+': 0.12,  # 3 or more entity matches
    })

    # Keyword overlap score thresholds (match count -> score)
    keyword_overlap_thresholds: Dict[str, float] = field(default_factory=lambda: {
        '1': 0.35,
        '2': 0.55,
        '3': 0.75,
        '4': 0.90,
        '5+': 1.0,
    })

    # Transcript quality thresholds
    transcript_quality_high: float = 0.8   # Score above this = high quality
    transcript_quality_medium: float = 0.5  # Score above this = medium quality
    transcript_min_words_good: int = 50     # Word count for "good" quality
    transcript_min_words_medium: int = 20   # Word count for "medium" quality

    # Multimodal default weights (must sum to 1.0)
    multimodal_default_weights: Dict[str, float] = field(default_factory=lambda: {
        'text_embedding': 0.40,
        'keyword_overlap': 0.25,
        'entity_match': 0.20,
        'visual_description': 0.15,
    })

    # Semantic coherence thresholds
    semantic_coherence_smooth_threshold: float = 0.6  # Above this = smooth flow
    semantic_coherence_abrupt_threshold: float = 0.3  # Below this = abrupt transition
    semantic_coherence_smooth_boost: float = 0.03     # Boost for smooth flow
    semantic_coherence_abrupt_penalty: float = 0.05   # Penalty for abrupt transition

    # Adaptive confidence floor by chapter type (US-77-006)
    # Intro/conclusion segments are more important and get a lower floor
    # so they survive even with lower confidence rather than being floored out
    adaptive_confidence_floor_enabled: bool = True
    adaptive_confidence_floor: Dict[str, float] = field(default_factory=lambda: {
        'intro': 0.03,
        'conclusion': 0.03,
        'body': 0.05,
    })

    # Pool normalization constants
    pool_normalization_reference_size: int = 50   # Reference pool size
    pool_normalization_min_factor: float = 0.8    # Min normalization factor
    pool_normalization_max_factor: float = 1.2    # Max normalization factor
    pool_small_threshold: int = 10                # Pool "small" below this
    pool_large_threshold: int = 100               # Pool "large" above this
    pool_tight_margin_threshold: float = 0.05     # Top-2 score diff for "tight margin"

    # Compounding guard (US-84-002) — dampen cascading negative adjustments
    max_cumulative_negative_adjustment: float = -0.30  # Threshold for dampening remaining negatives
    compounding_dampening_factor: float = 0.50  # Multiply remaining negatives by this when threshold exceeded

    # Source stutter penalty (US-84-004) — penalize A-B-A video alternation pattern
    # Detects when current source matches 2-segments-ago but differs from previous segment
    # This creates a jarring visual ping-pong effect (e.g., videoA -> videoB -> videoA)
    source_stutter_penalty: float = 0.04  # Magnitude of penalty for A-B-A pattern

    # Duration ratio reward curve (US-84-007) — smooth curve replacing step-function
    # Near-perfect duration matches (ratio within reward_threshold of 1.0) get a boost
    duration_ratio_reward_threshold: float = 0.1  # Ratio deviation from 1.0 to qualify for reward (0.9-1.1)
    duration_ratio_reward_boost: float = 0.02  # Confidence boost for near-perfect duration match

    # Ambiguous pool detection (US-84-008) — flag when top candidates score nearly identically
    # When top-10 candidate similarity variance < this threshold, the match is flagged as ambiguous
    variance_warning_threshold: float = 0.02  # Variance below this triggers ambiguous_pool flag

    # Inter-track embedding diversity (US-84-009) — measure how different V1/V2/V3 really are
    # When avg pairwise cosine distance between V1/V2/V3 embeddings < this threshold,
    # the segment is flagged as low-diversity (alternatives look too similar)
    min_track_diversity_distance: float = 0.15  # Minimum avg pairwise cosine distance

    def __post_init__(self):
        # Convert dict keys to strings if loaded from YAML as ints
        if isinstance(self.entity_match_boosts, dict):
            self.entity_match_boosts = {str(k): v for k, v in self.entity_match_boosts.items()}
        if isinstance(self.keyword_overlap_thresholds, dict):
            self.keyword_overlap_thresholds = {str(k): v for k, v in self.keyword_overlap_thresholds.items()}


@dataclass
class ContextEnrichmentConfig:
    """Controls extraction of video metadata for context-aware matching.

    When enabled, video description, chapters, and tags from yt-dlp info_dict
    are extracted and stored alongside captions to enrich matching signals.
    """
    extract_video_description: bool = True   # Extract video description text
    extract_video_chapters: bool = True      # Extract chapter markers from video
    extract_video_tags: bool = True          # Extract video tags/keywords
    max_description_length: int = 500        # Truncate descriptions longer than this
    parse_description_chapters: bool = True  # Parse chapter timestamps from description text
    title_enriched_embeddings: bool = True   # Include title+description in embedding generation
    chapter_enriched_embeddings: bool = True  # Include chapter title in embedding text when available
    description_enriched_embeddings: bool = True  # Append top description keywords to embedding text

    def __post_init__(self):
        import logging
        logger = logging.getLogger(__name__)

        # ValueError for impossible values (negative length)
        if self.max_description_length < 0:
            raise ValueError(
                f"ContextEnrichmentConfig.max_description_length="
                f"{self.max_description_length} is negative. "
                f"Must be >= 0. Check matching.context_enrichment."
                f"max_description_length in config.yaml"
            )

        # Warn + clamp for soft limit (unreasonably large)
        if self.max_description_length > 10000:
            logger.warning(
                "ContextEnrichmentConfig.max_description_length=%s exceeds "
                "reasonable limit of 10000, clamped to 10000",
                self.max_description_length,
            )
            self.max_description_length = 10000


@dataclass
class ChapterGroupingConfig:
    """Chapter-level source consistency settings (US-70-011).

    When segments fall within the same voiceover chapter, using clips from the
    same video source is desirable (topical coherence) rather than penalizable.
    This config controls the consistency boost and penalty suppression.
    """
    enabled: bool = True  # Enable chapter-level source grouping
    source_consistency_boost: float = 0.03  # Boost when same source within chapter
    coherence_penalty_threshold: int = 5  # Max unique sources per chapter before penalty
    relevance_boost_weight: float = 0.1  # Weight for cross-chapter relevance boost (US-71-005)
    min_source_diversity: int = 2  # Min unique sources per chapter (US-77-007); 1 = disabled
    chapter_topic_match_boost: List[float] = field(default_factory=lambda: [0.05, 0.15])  # [min, max] boost range for topic match
    chapter_topic_mismatch_penalty: float = -0.10  # Penalty when video topic doesn't match chapter (must be negative)

    def __post_init__(self):
        import logging
        logger = logging.getLogger(__name__)

        # ValueError for impossible values
        if self.coherence_penalty_threshold < 1:
            raise ValueError(
                f"ChapterGroupingConfig.coherence_penalty_threshold="
                f"{self.coherence_penalty_threshold} must be >= 1. "
                f"Check matching.chapter_grouping.coherence_penalty_threshold "
                f"in config.yaml"
            )

        # chapter_topic_match_boost: must be list of exactly 2 floats with min <= max
        if not isinstance(self.chapter_topic_match_boost, list) or len(self.chapter_topic_match_boost) != 2:
            raise ValueError(
                f"ChapterGroupingConfig.chapter_topic_match_boost must be a list of "
                f"exactly 2 floats [min, max], got {self.chapter_topic_match_boost}"
            )
        try:
            boost_min, boost_max = float(self.chapter_topic_match_boost[0]), float(self.chapter_topic_match_boost[1])
        except (TypeError, ValueError):
            raise ValueError(
                f"ChapterGroupingConfig.chapter_topic_match_boost values must be numeric, "
                f"got {self.chapter_topic_match_boost}"
            )
        if boost_min > boost_max:
            raise ValueError(
                f"ChapterGroupingConfig.chapter_topic_match_boost min ({boost_min}) "
                f"must be <= max ({boost_max}). Check matching.chapter_grouping."
                f"chapter_topic_match_boost in config.yaml"
            )

        # chapter_topic_mismatch_penalty: must be negative (it's a penalty)
        if self.chapter_topic_mismatch_penalty >= 0:
            raise ValueError(
                f"ChapterGroupingConfig.chapter_topic_mismatch_penalty="
                f"{self.chapter_topic_mismatch_penalty} must be negative (it's a penalty). "
                f"Check matching.chapter_grouping.chapter_topic_mismatch_penalty in config.yaml"
            )

        # relevance_boost_weight: ValueError if negative, warn+clamp if > 1.0
        if self.relevance_boost_weight < 0.0:
            raise ValueError(
                f"ChapterGroupingConfig.relevance_boost_weight="
                f"{self.relevance_boost_weight} is negative. "
                f"Must be in range [0.0, 1.0]. Check matching.chapter_grouping."
                f"relevance_boost_weight in config.yaml"
            )
        if self.relevance_boost_weight > 1.0:
            logger.warning(
                "ChapterGroupingConfig.relevance_boost_weight=%s exceeds 1.0, "
                "clamped to 1.0",
                self.relevance_boost_weight,
            )
            self.relevance_boost_weight = 1.0


@dataclass
class TieredCaptionPenalties:
    """Configurable thresholds for tiered caption quality penalties (US-78-008).

    Graduated penalties for specific quality issues detected on captions.
    Penalties stack but are capped at max_caption_penalty.
    Different projects may need different weights (e.g., music channels vs documentaries).
    """
    auto_generated_penalty: float = -0.05   # Penalty for auto-generated captions
    low_quality_penalty: float = -0.08      # Penalty for low quality captions
    missing_timing_penalty: float = -0.03   # Penalty for missing/poor timing data
    max_caption_penalty: float = -0.12      # Maximum combined caption penalty (cap)


@dataclass
class MatchingConfig:
    """Matching engine settings

    Chain-of-thought: Two-stage matching optimizes cost vs quality
    Reasoning: Embedding search cheap (FAISS), LLM expensive
    Decision: Retrieve 20 candidates → rerank top 5 with LLM
    """
    # Confidence thresholds
    min_confidence: float = 0.7
    high_confidence_threshold: float = 0.85  # Skip LLM if above
    skip_llm_threshold: float = 0.85  # Alias for high_confidence_threshold (for matching.py compatibility)
    low_confidence_threshold: float = 0.5    # Use secondary LLM if below
    ambiguous_threshold: float = 0.6  # Use secondary LLM if confidence < this
    confidence_threshold: float = 0.5  # Legacy alias for min_confidence

    # Adaptive threshold (adjusts skip_llm_threshold based on voiceover characteristics)
    # Short voiceover (<20 chars): +0.05 threshold (harder to match, require higher confidence)
    # Low candidate variance (<0.05): -0.05 threshold (clear winner, can accept lower)
    adaptive_threshold_enabled: bool = True

    # Negative sampling for LLM reranking
    # When enabled, includes a "unlikely match" sample from bottom 25% of candidates
    # This helps LLMs calibrate confidence by showing what a poor match looks like
    negative_sampling_enabled: bool = True

    # Clip reuse prevention
    max_clip_reuse: int = 1
    reuse_penalty: float = 0.5
    smart_reuse: bool = True
    sequential_when_reuse: bool = True

    # Consecutive same-source penalty (US-63-009)
    # Penalizes using the same video source in adjacent segments to improve visual variety
    # Penalty stacks: 1st repeat = 1x penalty, 2nd repeat = 2x penalty, etc.
    consecutive_source_penalty: float = 0.1  # Penalty per consecutive same-source match
    max_consecutive_same_source: int = 3  # Hard cap - block source after N consecutive uses

    # Global clip deduplication (hard block mode)
    # When True, same clip can NEVER appear twice anywhere in timeline (P1 requirement)
    clip_hard_block: bool = True

    # Source file reuse prevention (limits how many times any segment from same video can be used)
    max_source_file_reuse: int = 3  # 0 = unlimited, 3 = max 3 clips from same video
    source_file_penalty: float = 0.05  # Penalty per use after reaching half the max

    # Two-stage matching optimization
    embedding_candidates: int = 50  # Retrieve from FAISS (need 50+ for V1-V6)
    llm_rerank_candidates: int = 5  # Send to LLM
    top_k_candidates: int = 10  # Legacy alias

    # Source diversity in embedding search (US-77-009)
    max_candidates_per_source: int = 3  # Max candidates from same video_id (0 = disabled)

    # Pre-fetch diversity multiplier (US-84-005)
    # Fetch k*multiplier from FAISS before diversity filtering to avoid missing
    # diverse candidates ranked beyond the initial k when top results are dominated
    pre_fetch_multiplier: int = 2  # Multiplier for FAISS pre-fetch (1 = disabled)

    # Context window
    context_window: int = 2  # Consider N segments before/after

    # Duration scoring
    duration_scoring_enabled: bool = True
    ideal_speed_min: float = 0.85
    ideal_speed_max: float = 1.15
    soft_speed_min: float = 0.70
    soft_speed_max: float = 1.50
    duration_penalty_factor: float = 0.1

    # Tuple versions for matching.py compatibility
    @property
    def ideal_speed_range(self) -> tuple:
        return (self.ideal_speed_min, self.ideal_speed_max)

    @property
    def soft_penalty_range(self) -> tuple:
        return (self.soft_speed_min, self.soft_speed_max)

    # Keyword/entity boosting
    keyword_boost: float = 0.05
    entity_boost: float = 0.08

    # US-63-011: Named entity match boost
    # When voiceover text contains named entities (people, places, organizations)
    # and video caption contains matching entity, apply this confidence boost
    entity_match_boost: float = 0.1  # Default boost for entity matches

    # LLM providers (tiered: primary → secondary → local)
    primary_provider: str = "gemini"
    secondary_provider: str = "anthropic"
    local_provider: str = "ollama"
    use_local_for_review: bool = True

    # Model names
    gemini_model: str = "gemini-2.0-flash"
    anthropic_model: str = "claude-3-haiku-20240307"
    ollama_model: str = "llama3.2"
    local_llm_model: str = "llama3.2"  # Alias for ollama_model
    ollama_host: str = "http://localhost:11434"  # Ollama API host

    # Caching
    cache_llm_responses: bool = True
    cache_ttl_hours: int = 24

    # LLM reranker spread calibration (US-63-008)
    # Adjusts confidence based on candidate spread (top-1 vs top-2 similarity)
    llm_reranker_close_spread_threshold: float = 0.05  # Spread < this = ambiguous, reduce confidence
    llm_reranker_clear_winner_threshold: float = 0.20  # Spread > this = clear winner, boost confidence
    llm_reranker_close_spread_factor: float = 0.9  # Multiply confidence by this when close spread
    llm_reranker_clear_winner_factor: float = 1.1  # Multiply confidence by this when clear winner

    # Delta matching (only match new videos)
    delta_matching_enabled: bool = True  # Enable delta-aware matching
    force_rematch: bool = False  # Force rematch all videos (CLI override)
    rematch_improvement_threshold: float = 0.05  # Re-evaluate if new video > existing + this

    # Chapter/topic matching
    chapter_matching_enabled: bool = True  # Enable chapter-based topic filtering
    topic_mismatch_penalty: float = 0.15  # Confidence penalty for topic mismatch
    extract_video_topics: bool = True  # Extract topics from video transcripts
    min_topic_overlap: int = 1  # Minimum topic keywords that must match

    # B-roll boost (silent videos are valuable)
    # Boosts confidence for: 1) silent/B-roll videos, 2) scenes without faces when topic matches
    broll_boost: float = 0.2  # Confidence boost for B-roll videos (0.0-0.3)

    # Caption quality confidence adjustment (US-007, US-006, US-73-006)
    # Adjusts confidence based on caption quality: high, medium, low
    # Two modes available:
    #   1. Additive (legacy): high_boost/low_penalty add/subtract from confidence
    #   2. Multiplicative weights (US-006): confidence = raw * weight
    caption_quality_adjustment_enabled: bool = True  # Enable caption quality confidence adjustment

    # Legacy additive mode (US-007) - used when caption_quality_weights is None
    caption_quality_high_boost: float = 0.05  # Boost for high-quality human captions
    caption_quality_low_penalty: float = 0.1  # Penalty for low-quality/fallback captions

    # Multiplicative weights mode (US-006) - when set, overrides additive mode
    # Keys: 'high', 'medium', 'low'; Values: multiplier (0.0-1.0)
    # Example: {high: 1.0, medium: 0.9, low: 0.75}
    # adjusted = raw_confidence * weight
    caption_quality_weights: Optional[Dict[str, float]] = None  # None = use additive mode

    # Tiered caption quality penalties (US-73-006, US-78-008)
    # Graduated penalties for specific quality issues - stack up to max_caption_penalty
    # Legacy flat fields (backward compat) - prefer tiered_caption_penalties nested config
    caption_penalty_auto_generated: float = -0.05  # Penalty for auto-generated captions
    caption_penalty_low_quality: float = -0.08     # Penalty for low quality captions
    caption_penalty_missing_timing: float = -0.03  # Penalty for missing/poor timing data
    max_caption_penalty: float = -0.12             # Maximum combined caption penalty (cap)

    # Nested config (US-78-008) - when set, overrides flat fields above
    tiered_caption_penalties: TieredCaptionPenalties = None

    # Language confidence penalty (US-73-012)
    # Penalty multiplier for low language_confidence captions (auto-translated)
    # Final penalty = (1.0 - language_confidence) * language_confidence_penalty
    # Default 0.0 = disabled; reasonable value when enabled: 0.1
    language_confidence_penalty: float = 0.0

    # Caption timing penalty (US-008 Sprint 7)
    # Penalizes matches when caption timing is poor (low coverage, exceeds video duration)
    # Formula: adjusted = raw * quality_weight * timing_penalty
    # timing_penalty = 1.0 - (exceeds_ratio * 0.3) - ((1 - coverage_ratio) * 0.2)
    # Example: 50% coverage, 20% exceeds -> timing_penalty = 0.84 (~16% penalty)
    apply_timing_penalty: bool = True  # Enable timing-based confidence penalty

    # B-roll preference (scene-level face detection)
    # When topic matches, prefer scenes without faces (B-roll) over talking heads
    prefer_broll_when_topic_matches: bool = True  # Enable B-roll preference
    broll_face_threshold: float = 0.3  # face_score < this = B-roll (no faces)

    # Temporal coherence (adjacent segment scoring)
    # Encourages visual continuity by scoring clips based on similarity to adjacent segments
    temporal_coherence_enabled: bool = True  # Enable temporal coherence scoring
    temporal_coherence_same_source_boost: float = 0.05  # Boost for clips from same source as adjacent
    temporal_coherence_context_switch_penalty: float = 0.05  # Penalty for jarring context switches

    # Fallback matching (for edge cases when primary matching fails)
    # Triggers when primary matching returns confidence < fallback_trigger_threshold
    # Provides 3 levels: keyword-only (0.7), visual-description (0.5), generic-broll (0.3)
    fallback_matching_enabled: bool = True
    fallback_trigger_threshold: float = 0.4  # Trigger fallback when confidence below this

    # Transcript quality scoring (adjust embedding weight based on transcript quality)
    # Low-quality transcripts (short, incoherent, repetitive) are less reliable for matching
    # When enabled, reduces embedding weight by 20% for low-quality transcripts
    transcript_quality_weight: bool = True  # Enable transcript quality-based weight adjustment

    # Pool size normalization (adjusts confidence based on candidate pool size)
    # Small pools (<10): Boost confidence when top match is clear
    # Large pools (>100): Reduce confidence when margins are tight
    # Formula: sqrt(pool_size/50) capped at [0.8, 1.2]
    pool_normalization_enabled: bool = True  # Enable confidence normalization by pool size

    # Chain-of-thought prompting for LLM matching
    # When enabled, uses structured 4-step reasoning prompt:
    # 1. Identify voiceover themes, 2. List video elements
    # 3. Evaluate rubric (visual 30%, topic 40%, keyword 20%, flow 10%)
    # 4. Compute weighted final score
    # Produces more reliable and explainable match decisions
    chain_of_thought_enabled: bool = True  # Enable chain-of-thought structured prompting

    # Early termination for obvious high-confidence matches
    # When enabled, skips LLM when match is obviously good:
    # - Embedding similarity > 0.9
    # - At least 3 keywords match
    # - Same named entity found in both voiceover and video
    # Returns boosted confidence (min 0.92) with reason 'obvious_match_early_termination'
    obvious_match_enabled: bool = True  # Enable early termination for obvious matches
    obvious_match_min_similarity: float = 0.9  # Minimum embedding similarity for obvious match
    obvious_match_min_keywords: int = 3  # Minimum matched keywords for obvious match
    obvious_match_min_confidence: float = 0.92  # Minimum confidence for obvious matches

    # Multi-modal similarity weighting
    # When enabled, combines multiple similarity signals with weighted fusion:
    # - text_embedding (40%): Semantic similarity from embeddings
    # - keyword_overlap (25%): Matched keywords between voiceover and video
    # - entity_match (20%): Named entity overlap (people, places, organizations)
    # - visual_description (15%): Visual scene description similarity
    # Replaces simple additive boosting with weighted multi-modal scoring
    multimodal_enabled: bool = True  # Enable multi-modal similarity weighting
    multimodal_weights: Optional[Dict[str, float]] = None  # Custom weights dict (defaults used if None)

    # Semantic coherence (topic flow between adjacent segments)
    # When enabled, evaluates embedding similarity between current match candidate
    # and the previous segment's match to ensure smooth topic transitions.
    # - Smooth flow (similarity > 0.6): +0.03 boost (good continuity)
    # - Abrupt flow (similarity < 0.3): -0.05 penalty (jarring transition)
    # - Neutral (0.3 - 0.6): no adjustment
    semantic_coherence_enabled: bool = True  # Enable semantic coherence scoring

    # Explanation confidence validation
    # When enabled, cross-checks that LLM explanation keywords appear in actual
    # voiceover/video text. When less than 50% of keywords are verifiable,
    # confidence is downgraded by 0.1 to penalize "hallucinated" explanations.
    explanation_validation_enabled: bool = True  # Enable explanation keyword verification

    # Location-aware matching (for travel/location content)
    location_matching: LocationMatchingConfig = None

    # Enhanced chapter detection
    chapter_detection: ChapterDetectionConfig = None

    # Scoring thresholds (US-53-002)
    scoring: MatchingScoringConfig = None

    # Context enrichment (US-70-004)
    context_enrichment: ContextEnrichmentConfig = None

    # Chapter-level source grouping (US-70-011)
    chapter_grouping: ChapterGroupingConfig = None

    def __post_init__(self):
        import logging
        logger = logging.getLogger(__name__)

        # Validate confidence thresholds (must be 0.0-1.0)
        # ValueError for impossible values (negative); warn+clamp for soft limits (>1.0)
        confidence_fields = [
            'min_confidence', 'high_confidence_threshold',
            'low_confidence_threshold', 'ambiguous_threshold',
            'skip_llm_threshold', 'confidence_threshold',
        ]
        for field_name in confidence_fields:
            value = getattr(self, field_name)
            if value < 0.0:
                raise ValueError(
                    f"MatchingConfig.{field_name}={value} is negative. "
                    f"Confidence thresholds must be in range [0.0, 1.0]. "
                    f"Check matching.{field_name} in config.yaml"
                )
            if value > 1.0:
                logger.warning(
                    "MatchingConfig.%s=%s exceeds 1.0, clamped to 1.0",
                    field_name, value,
                )
                setattr(self, field_name, 1.0)

        # Validate positive integer fields — ValueError for impossible values (<= 0)
        positive_int_fields = [
            'embedding_candidates',
            'llm_rerank_candidates',
            'top_k_candidates',
            'context_window',
        ]
        for field_name in positive_int_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(
                    f"MatchingConfig.{field_name}={value} must be a positive integer. "
                    f"Check matching.{field_name} in config.yaml"
                )

        # Nested dataclass conversion
        if self.location_matching is None:
            self.location_matching = LocationMatchingConfig()
        elif isinstance(self.location_matching, dict):
            self.location_matching = LocationMatchingConfig(**self.location_matching)

        if self.chapter_detection is None:
            self.chapter_detection = ChapterDetectionConfig()
        elif isinstance(self.chapter_detection, dict):
            self.chapter_detection = ChapterDetectionConfig(**self.chapter_detection)

        if self.scoring is None:
            self.scoring = MatchingScoringConfig()
        elif isinstance(self.scoring, dict):
            self.scoring = MatchingScoringConfig(**self.scoring)

        if self.context_enrichment is None:
            self.context_enrichment = ContextEnrichmentConfig()
        elif isinstance(self.context_enrichment, dict):
            self.context_enrichment = ContextEnrichmentConfig(**self.context_enrichment)

        if self.chapter_grouping is None:
            self.chapter_grouping = ChapterGroupingConfig()
        elif isinstance(self.chapter_grouping, dict):
            self.chapter_grouping = ChapterGroupingConfig(**self.chapter_grouping)

        # Tiered caption penalties (US-78-008)
        # When nested config provided (from YAML), it takes precedence and syncs to flat fields.
        # When only flat fields are set (backward compat), build nested config from them.
        if isinstance(self.tiered_caption_penalties, dict):
            self.tiered_caption_penalties = TieredCaptionPenalties(**self.tiered_caption_penalties)
            # Sync nested -> flat for backward compat with scoring.py getattr
            self.caption_penalty_auto_generated = self.tiered_caption_penalties.auto_generated_penalty
            self.caption_penalty_low_quality = self.tiered_caption_penalties.low_quality_penalty
            self.caption_penalty_missing_timing = self.tiered_caption_penalties.missing_timing_penalty
            self.max_caption_penalty = self.tiered_caption_penalties.max_caption_penalty
        elif self.tiered_caption_penalties is None:
            # Build nested config from flat fields (backward compat or constructor override)
            self.tiered_caption_penalties = TieredCaptionPenalties(
                auto_generated_penalty=self.caption_penalty_auto_generated,
                low_quality_penalty=self.caption_penalty_low_quality,
                missing_timing_penalty=self.caption_penalty_missing_timing,
                max_caption_penalty=self.max_caption_penalty,
            )
