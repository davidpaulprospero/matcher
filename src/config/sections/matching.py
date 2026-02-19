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
    'ListicleTopicConfig',
    'ListicleBoundaryConfig',
    'VoiceoverTopicConfig',
    'ContextPriorityWeights',
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
class MatchingScoringConfig:
    """Scoring thresholds and weights for match confidence calculations.

    Extracted from hardcoded constants in src/matching/scoring.py (US-53-002).
    All fields have defaults matching the original hardcoded values.
    """

    # Confidence floor and warning (applied in MatchScoring.apply_all_adjustments)
    confidence_floor: float = 0.05  # Minimum confidence after all penalties
    low_confidence_warning_threshold: float = 0.15  # Warn when penalized below this

    # US-135-004: Temporal overlap weighting for chapter alignment
    # Weight for combining keyword similarity with temporal overlap percentage
    # Score formula: keyword_similarity * (1 - temporal_weight) + overlap_pct * temporal_weight
    temporal_overlap_weight: float = 0.3  # Weight for temporal overlap (0.0-1.0)
    # Minimum overlap threshold below which chapter alignment is not applied
    minimum_overlap_threshold: float = 0.3  # Minimum 30% overlap required

    # Chapter alignment confidence (US-105-004)
    # Minimum confidence threshold for chapter-aligned matches to receive boost
    # Matches with chapter alignment quality >= this threshold get the chapter_alignment_boost
    chapter_match_confidence_min: float = 0.6  # Minimum confidence for chapter-aligned boost

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

    # US-134-008: Enhanced semantic coherence config
    semantic_window_size: int = 3  # Number of segments before/after to consider
    semantic_coherence_min_threshold: float = 0.5  # Minimum average coherence for boost
    topic_drift_detection_enabled: bool = True  # Detect topic shifts within chapters

    # US-141-010: Visual-textual context fusion scoring
    # Combines visual description similarity with text metadata (title/description/tags)
    # to improve confidence calibration when both signals are available
    visual_text_fusion_enabled: bool = True  # Enable visual-text fusion scoring
    visual_text_weight: float = 0.20  # Weight for visual component in fusion (0.0-1.0)

    # Adaptive confidence floor by chapter type (US-77-006, US-117-007)
    # Different chapter types get different floors:
    # - intro: First chapter (0.10) - important opening
    # - chapter: Middle chapters (0.05) - default
    # - outro: Last chapter (0.08) - closing content
    # - standalone: Single chapter (0.15) - self-contained segments
    adaptive_confidence_floor_enabled: bool = True
    adaptive_confidence_floor: Dict[str, float] = field(default_factory=lambda: {
        'intro': 0.10,
        'chapter': 0.05,
        'outro': 0.08,
        'standalone': 0.15,
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

    # Source channel coherence boost (US-95-006) — reward videos from same source channel
    # When current video is from same YouTube channel as previous match, apply a boost
    # This rewards consistent visual style/theme across matched segments
    source_channel_coherence_boost: float = 0.05  # Boost for same-channel videos

    # Topic alignment weight (US-95-007) — boost confidence when voiceover topics match video topics
    # Voiceover topic extraction aligns with video chapter topics; this boosts confidence
    # for matched segments where topics align
    topic_alignment_weight: float = 0.1  # Default boost for topic alignment

    # Duration ratio reward curve (US-84-007) — smooth curve replacing step-function
    # Near-perfect duration matches (ratio within reward_threshold of 1.0) get a boost
    duration_ratio_reward_threshold: float = 0.1  # Ratio deviation from 1.0 to qualify for reward (0.9-1.1)
    duration_ratio_reward_boost: float = 0.02  # Confidence boost for near-perfect duration match

    # US-134-006: Duration context boost — higher confidence when durations are similar
    # When video/voiceover duration ratio is within optimal range, apply a confidence boost
    # When ratio is outside optimal but within acceptable range, apply a penalty
    duration_context_boost_enabled: bool = False  # Enable duration context scoring
    duration_optimal_ratio_range: List[float] = field(default_factory=lambda: [0.8, 1.2])  # [min, max] for boost
    duration_boost_max: float = 0.05  # Max boost when ratio is optimal
    duration_mismatch_penalty_max: float = 0.10  # Max penalty when ratio is far from optimal

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

        # US-134-006: Validate duration context config
        # duration_optimal_ratio_range: must be list of 2 floats with min <= max
        if not isinstance(self.duration_optimal_ratio_range, list) or len(self.duration_optimal_ratio_range) != 2:
            raise ValueError(
                f"MatchingScoringConfig.duration_optimal_ratio_range must be a list of "
                f"exactly 2 floats [min, max], got {self.duration_optimal_ratio_range}"
            )
        try:
            opt_min, opt_max = float(self.duration_optimal_ratio_range[0]), float(self.duration_optimal_ratio_range[1])
        except (TypeError, ValueError):
            raise ValueError(
                f"MatchingScoringConfig.duration_optimal_ratio_range values must be numeric, "
                f"got {self.duration_optimal_ratio_range}"
            )
        if opt_min > opt_max:
            raise ValueError(
                f"MatchingScoringConfig.duration_optimal_ratio_range min ({opt_min}) "
                f"must be <= max ({opt_max}). Check matching.scoring.duration_optimal_ratio_range "
                f"in config.yaml"
            )

        # Validate duration_boost_max and duration_mismatch_penalty_max are non-negative
        if self.duration_boost_max < 0:
            raise ValueError(
                f"MatchingScoringConfig.duration_boost_max={self.duration_boost_max} must be >= 0. "
                f"Check matching.scoring.duration_boost_max in config.yaml"
            )
        if self.duration_mismatch_penalty_max < 0:
            raise ValueError(
                f"MatchingScoringConfig.duration_mismatch_penalty_max={self.duration_mismatch_penalty_max} must be >= 0. "
                f"Check matching.scoring.duration_mismatch_penalty_max in config.yaml"
            )

    # View count as quality signal (US-134-009)
    # Uses logarithmic scaling to reduce outlier impact from viral videos
    view_count_context_weight: float = 0.02  # Weight for view count as context signal (0 = disabled)
    view_count_boost_threshold: int = 1000000  # View count threshold for boost (1M views)
    view_count_log_scale: bool = True  # Use log scale to reduce outlier impact
    view_count_quality_threshold: int = 1000  # Minimum views for quality signal


@dataclass
class ContextEnrichmentConfig:
    """Controls extraction of video metadata for context-aware matching.

    When enabled, video description, chapters, and tags from yt-dlp info_dict
    are extracted and stored alongside captions to enrich matching signals.

    US-111-008: Enrichment factors control weighted combination of metadata signals
    in embedding text construction. Factors should sum to <=1.0 for balanced weighting.
    """
    extract_video_description: bool = True   # Extract video description text
    extract_video_chapters: bool = True      # Extract chapter markers from video
    extract_video_tags: bool = True          # Extract video tags/keywords
    max_description_length: int = 500        # Truncate descriptions longer than this
    parse_description_chapters: bool = True  # Parse chapter timestamps from description text

    # US-135-008: Chapter extraction fallback strategy
    # Controls how to use description chapters when metadata chapters are missing/insufficient:
    # - 'disabled': Don't use description chapters (current default behavior)
    # - 'description': Use description chapters when metadata chapters missing (default)
    # - 'always': Merge metadata + description chapters (metadata takes precedence)
    chapter_extraction_fallback: str = 'description'

    title_enriched_embeddings: bool = True   # Include title+description in embedding generation
    chapter_enriched_embeddings: bool = True  # Include chapter title in embedding text when available
    description_enriched_embeddings: bool = True  # Append top description keywords to embedding text
    embed_channel_context: bool = True  # Include channel name in embedding text when available
    embed_channel_reputation: bool = True  # Include channel subscriber count and reputation in embedding text (US-134-004)
    tag_boost_enabled: bool = True  # Enable tag-based keyword boost in scoring (US-127-003)

    # US-141-007: Tag relevance scoring with position weighting
    # Controls how tag relevance is computed:
    # - tag_position_decay: decay factor for tags by position (first tags more important)
    # - tag_frequency_weight: weight for frequency-based scoring (0-1)
    tag_position_decay: float = 0.9  # Each position multiplies relevance by this factor
    tag_frequency_weight: float = 0.15  # Weight for frequency-based component

    # US-141-006: Description summarization for context enrichment
    # When enabled, uses LLM to extract most relevant description snippets for matching
    # instead of using raw truncated descriptions
    description_summarization_enabled: bool = False  # Enable LLM-based description summarization
    summary_max_words: int = 50  # Maximum words in LLM-generated summary

    # US-111-008: Multi-signal embedding context enrichment factors
    # These control how much weight each metadata signal has in embedding text
    # US-134-003: tags_enrichment_factor increased from 0.2 to 0.25 based on improved tag filtering
    description_enrichment_factor: float = 0.3  # Weight for description keywords (0-1)
    tags_enrichment_factor: float = 0.25  # Weight for video tags (0-1)
    chapters_enrichment_factor: float = 0.3  # Weight for chapter titles (0-1)

    # US-126-005: Constraint - enrichment factors must sum to <= 1.0
    VALIDATION_SUM_LIMIT: float = 1.0
    # Warn when sum is below this threshold (under-utilizing context signals)
    UNDER_UTILIZATION_THRESHOLD: float = 0.5

    def __post_init__(self):
        import logging
        from ..schema_validation import ConfigValidationError

        logger = logging.getLogger(__name__)

        # US-127-010: Validate boolean fields are actually booleans
        # YAML loading can produce strings ("true") or ints (1) instead of bools
        boolean_fields = [
            'extract_video_description',
            'extract_video_chapters',
            'extract_video_tags',
            'parse_description_chapters',
            'title_enriched_embeddings',
            'chapter_enriched_embeddings',
            'description_enriched_embeddings',
            'embed_channel_context',
            'embed_channel_reputation',
            'tag_boost_enabled',
            'description_summarization_enabled',  # US-141-006
        ]
        # US-141-007: Validate new tag scoring config fields
        if self.tag_position_decay is not None:
            if not isinstance(self.tag_position_decay, (int, float)):
                raise ValueError(
                    f"ContextEnrichmentConfig.tag_position_decay must be a number, "
                    f"got {type(self.tag_position_decay).__name__}: {self.tag_position_decay!r}"
                )
            if not 0.0 <= self.tag_position_decay <= 1.0:
                raise ValueError(
                    f"ContextEnrichmentConfig.tag_position_decay must be between 0.0 and 1.0, "
                    f"got {self.tag_position_decay}"
                )
        if self.tag_frequency_weight is not None:
            if not isinstance(self.tag_frequency_weight, (int, float)):
                raise ValueError(
                    f"ContextEnrichmentConfig.tag_frequency_weight must be a number, "
                    f"got {type(self.tag_frequency_weight).__name__}: {self.tag_frequency_weight!r}"
                )
            if not 0.0 <= self.tag_frequency_weight <= 1.0:
                raise ValueError(
                    f"ContextEnrichmentConfig.tag_frequency_weight must be between 0.0 and 1.0, "
                    f"got {self.tag_frequency_weight}"
                )

        for field_name in boolean_fields:
            value = getattr(self, field_name, None)
            if not isinstance(value, bool):
                raise ValueError(
                    f"ContextEnrichmentConfig.{field_name} must be a boolean, "
                    f"got {type(value).__name__}: {value!r}. "
                    f"Check matching.context_enrichment.{field_name} in config.yaml"
                )

        # US-135-008: Validate chapter_extraction_fallback is one of valid options
        valid_fallback_options = {'disabled', 'description', 'always'}
        if self.chapter_extraction_fallback not in valid_fallback_options:
            raise ValueError(
                f"ContextEnrichmentConfig.chapter_extraction_fallback must be one of "
                f"{valid_fallback_options}, got '{self.chapter_extraction_fallback}'. "
                f"Check matching.context_enrichment.chapter_extraction_fallback in config.yaml"
            )

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

        # US-141-006: Validate summary_max_words
        if self.summary_max_words <= 0:
            raise ValueError(
                f"ContextEnrichmentConfig.summary_max_words="
                f"{self.summary_max_words} must be > 0. "
                f"Check matching.context_enrichment.summary_max_words in config.yaml"
            )

        # US-111-008: Validate enrichment factors are in valid range
        for factor_name in ['description_enrichment_factor', 'tags_enrichment_factor', 'chapters_enrichment_factor']:
            factor_value = getattr(self, factor_name)
            if factor_value < 0 or factor_value > 1:
                raise ValueError(
                    f"ContextEnrichmentConfig.{factor_name}={factor_value} "
                    f"must be between 0 and 1. Check matching.context_enrichment.{factor_name} in config.yaml"
                )

        # US-126-005: Validate enrichment factors sum <= 1.0
        total_factor = (
            self.description_enrichment_factor +
            self.tags_enrichment_factor +
            self.chapters_enrichment_factor
        )

        # Raise error if factors sum exceeds 1.0
        if total_factor > self.VALIDATION_SUM_LIMIT:
            raise ConfigValidationError(
                f"ContextEnrichmentConfig enrichment factors sum to {total_factor:.2f} (>1.0). "
                f"Factors must sum to <= 1.0 for balanced weighting. "
                f"Current values: description={self.description_enrichment_factor}, "
                f"tags={self.tags_enrichment_factor}, chapters={self.chapters_enrichment_factor}. "
                f"Check matching.context_enrichment in config.yaml"
            )

        # Warn if factors sum is below threshold (under-utilizing context signals)
        if total_factor < self.UNDER_UTILIZATION_THRESHOLD:
            logger.warning(
                "ContextEnrichmentConfig enrichment factors sum to %.2f (<%.2f). "
                "Context signals may be under-utilized. "
                "Consider increasing: description=%.2f, tags=%.2f, chapters=%.2f",
                total_factor,
                self.UNDER_UTILIZATION_THRESHOLD,
                self.description_enrichment_factor,
                self.tags_enrichment_factor,
                self.chapters_enrichment_factor,
            )


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

    # Multi-chapter segment assignment (US-105-009, US-135-012)
    # Strategy for assigning segments that span multiple chapters:
    # - 'first': Assign to the first chapter the segment overlaps with
    # - 'split': Assign to chapter where segment's midpoint falls
    # - 'best_match': Assign to chapter with greatest overlap duration (default)
    # - 'adaptive': Choose best strategy based on segment vs chapter duration ratio (US-135-012)
    multi_chapter_assignment_strategy: str = 'best_match'

    # Adaptive strategy thresholds (US-135-012)
    # Ratio of segment duration to chapter duration determines strategy:
    # - short_threshold: If ratio < short_threshold, use 'first' (segment clearly belongs)
    # - long_threshold: If ratio > long_threshold, use 'split' (spans chapters)
    # - Otherwise: use 'best_match' (medium segment, overlap approach)
    adaptive_short_threshold: float = 0.25  # < 25% of chapter = short
    adaptive_long_threshold: float = 0.75   # > 75% of chapter = long

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

        # multi_chapter_assignment_strategy: must be one of 'first', 'split', 'best_match', 'adaptive'
        valid_strategies = {'first', 'split', 'best_match', 'adaptive'}
        if self.multi_chapter_assignment_strategy not in valid_strategies:
            raise ValueError(
                f"ChapterGroupingConfig.multi_chapter_assignment_strategy="
                f"'{self.multi_chapter_assignment_strategy}' must be one of {valid_strategies}. "
                f"Check matching.chapter_grouping.multi_chapter_assignment_strategy "
                f"in config.yaml"
            )

        # adaptive_short_threshold: must be in [0.0, 1.0)
        if not isinstance(self.adaptive_short_threshold, (int, float)) or not (0.0 <= self.adaptive_short_threshold < 1.0):
            raise ValueError(
                f"ChapterGroupingConfig.adaptive_short_threshold="
                f"{self.adaptive_short_threshold} must be a number in [0.0, 1.0). "
                f"Check matching.chapter_grouping.adaptive_short_threshold in config.yaml"
            )

        # adaptive_long_threshold: must be in (0.0, 1.0]
        if not isinstance(self.adaptive_long_threshold, (int, float)) or not (0.0 < self.adaptive_long_threshold <= 1.0):
            raise ValueError(
                f"ChapterGroupingConfig.adaptive_long_threshold="
                f"{self.adaptive_long_threshold} must be a number in (0.0, 1.0]. "
                f"Check matching.chapter_grouping.adaptive_long_threshold in config.yaml"
            )

        # long_threshold must be > short_threshold
        if self.adaptive_long_threshold <= self.adaptive_short_threshold:
            raise ValueError(
                f"ChapterGroupingConfig.adaptive_long_threshold="
                f"{self.adaptive_long_threshold} must be > adaptive_short_threshold "
                f"({self.adaptive_short_threshold}). "
                f"Check matching.chapter_grouping in config.yaml"
            )


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
class ListicleTopicConfig:
    """Listicle topic extraction settings (US-105-005, US-135-003).

    Controls whether to use LLM or embeddings for better keyword extraction when
    simple keyword extraction yields insufficient results.
    """
    use_llm_topic_extraction: bool = False  # Use LLM when simple extraction yields <3 keywords
    min_keywords_for_simple: int = 3  # Minimum keywords needed before LLM fallback triggers
    # Embedding-based enhancement (US-135-003)
    use_embedding_topic_extraction: bool = False  # Use embeddings to find related keywords
    embedding_similarity_threshold: float = 0.6  # Minimum similarity for embedding-boosted keywords
    # Header language detection (US-135-010)
    # Set to list of language codes (en, es, fr, de, pt, it, ja) or ['auto'] for all
    # Empty list or None = all languages enabled
    header_lang_detection: List[str] = None  # Default: auto-detect all supported languages
    # Auto-correction for inconsistent numbering (US-140-004)
    # When true, normalizes mixed numbering formats (e.g., 'first, #3, third' -> '1st, 2nd, 3rd')
    auto_correction: bool = True  # Enable auto-correction of inconsistent numbering


@dataclass
class ListicleBoundaryConfig:
    """Listicle boundary pre-filtering settings (US-135-006).

    Controls whether to filter video candidates based on listicle group boundaries
    before running full matching. This can significantly reduce the number of
    candidates to process when listicle structure is detected.
    """
    # Strictness mode:
    # - 'strict': Only allow candidates from same listicle group
    # - 'relaxed': Allow candidates from same + adjacent groups
    # - 'disabled': No listicle-based filtering
    boundary_strictness: str = 'relaxed'  # Default to relaxed for backward compatibility
    # Minimum topic keyword overlap required for candidate to be considered
    # Only used when strictness is 'strict' or 'relaxed'
    min_topic_overlap: float = 0.2  # At least 20% topic overlap required
    # Enable fallback to all candidates when no matching candidates found
    fallback_on_empty: bool = True

    def __post_init__(self):
        # Validate boundary_strictness value
        valid_strictness = ['strict', 'relaxed', 'disabled']
        if self.boundary_strictness not in valid_strictness:
            raise ValueError(
                f"ListicleBoundaryConfig.boundary_strictness must be one of {valid_strictness}, "
                f"got '{self.boundary_strictness}'. Check matching.listicle_boundary.boundary_strictness "
                f"in config.yaml"
            )

        # Validate min_topic_overlap range
        if not 0.0 <= self.min_topic_overlap <= 1.0:
            raise ValueError(
                f"ListicleBoundaryConfig.min_topic_overlap must be in range [0.0, 1.0], "
                f"got {self.min_topic_overlap}. Check matching.listicle_boundary.min_topic_overlap "
                f"in config.yaml"
            )


@dataclass
class VoiceoverTopicConfig:
    """Voiceover segment topic extraction settings (US-111-002, US-126-006).

    Controls extraction of topics from voiceover segments using LLM
    for better context-aware matching.
    """
    enabled: bool = True  # Enable voiceover topic extraction
    min_topics: int = 3  # Minimum topics to extract per segment
    max_topics: int = 5  # Maximum topics to extract per segment (US-126-006)
    min_segment_length: int = 50  # Minimum text length to trigger extraction (shorter = skip)

    # Retry settings for LLM failures (US-126-006)
    retry_max_attempts: int = 3  # Maximum retry attempts for LLM failures
    retry_base_delay: float = 1.0  # Base delay in seconds for exponential backoff
    retry_max_delay: float = 10.0  # Maximum delay cap in seconds

    # Fallback to keyword extraction when LLM fails (US-126-006)
    fallback_to_keywords: bool = True  # Enable fallback to keyword extraction on LLM failure

    # Caching for topic extraction results (US-126-006)
    cache_enabled: bool = True  # Enable caching of topic extraction results

    # Context coherence settings (US-111-003)
    topic_coherence_enabled: bool = True  # Enable topic coherence scoring for context
    min_topic_similarity: float = 0.3  # Minimum topic similarity to consider segments coherent
    coherence_boost: float = 0.05  # Boost when adjacent segments have high topic similarity
    context_window_adjustment: bool = True  # Adjust context window size based on topic coherence
    max_context_segments: int = 4  # Maximum segments to include in context window


@dataclass
class ContextPriorityWeights:
    """US-111-007: Context priority weights for metadata signals in LLM prompts.

    Controls how much weight the LLM gives to different context signals
    when evaluating video candidates. Values should sum to 1.0.

    Attributes:
        title: Weight for title signal (default 0.35)
        description: Weight for description signal (default 0.30)
        tags: Weight for tags signal (default 0.20)
        chapters: Weight for chapters signal (default 0.15)
        adaptive_context_weights: Enable adaptive weighting based on available metadata (US-134-002)
    """

    title: float = 0.35
    description: float = 0.30
    tags: float = 0.20
    chapters: float = 0.15
    adaptive_context_weights: bool = False  # US-134-002: Adaptive weighting based on metadata availability

    # Tolerance for near-1.0 sums with auto-normalization
    # Accepts sums between 0.8 and 1.2 (auto-normalizes to 1.0)
    NORMALIZATION_TOLERANCE: float = 0.20

    def __post_init__(self):
        import logging
        from ..schema_validation import ConfigValidationError

        logger = logging.getLogger(__name__)

        # Check for empty weights (all default to 0)
        total = self.title + self.description + self.tags + self.chapters

        if total == 0:
            raise ConfigValidationError(
                "ContextPriorityWeights: All weights are zero. "
                "At least one weight must be non-zero. "
                "Check matching.context_priority_weights in config.yaml"
            )

        # Validate all weights are non-negative
        for name in ['title', 'description', 'tags', 'chapters']:
            value = getattr(self, name)
            if value < 0:
                raise ConfigValidationError(
                    f"ContextPriorityWeights.{name}={value} is negative. "
                    f"Weights must be non-negative. "
                    f"Check matching.context_priority_weights in config.yaml"
                )

        # Auto-normalize if within tolerance of 1.0
        # Tolerance: sums between 0.8 and 1.2 get auto-normalized to 1.0
        if (1.0 - self.NORMALIZATION_TOLERANCE) <= total <= (1.0 + self.NORMALIZATION_TOLERANCE):
            if abs(total - 1.0) > 0.001:  # Not effectively 1.0 (allow for float precision)
                logger.warning(
                    "ContextPriorityWeights: weights sum to %.2f, auto-normalizing to 1.0. "
                    "Original values: title=%.2f, description=%.2f, tags=%.2f, chapters=%.2f",
                    total, self.title, self.description, self.tags, self.chapters,
                )
                # Normalize weights
                self.title = self.title / total
                self.description = self.description / total
                self.tags = self.tags / total
                self.chapters = self.chapters / total
        elif abs(total - 1.0) > self.NORMALIZATION_TOLERANCE:
            # Outside tolerance - this is handled at the config schema level
            # but we log a warning for visibility
            logger.debug(
                "ContextPriorityWeights: weights sum to %.2f (tolerance=%.2f). "
                "Will be validated by schema.",
                total, self.NORMALIZATION_TOLERANCE,
            )


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

    # Retry settings for matching
    max_retries: int = 3  # Maximum retry attempts for matching

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

    # Cross-listicle source diversity penalty (US-135-009)
    # Penalizes using the same video source across different listicle items to improve variety
    # When multiple listicle items (different groups) use the same source consecutively
    listicle_diversity_penalty_enabled: bool = True  # Enable cross-listicle diversity penalty
    listicle_diversity_penalty_2_consecutive: float = 0.02  # Penalty for 2+ consecutive listicle items with same source
    listicle_diversity_penalty_3_plus: float = 0.05  # Additional penalty per repeat after 2
    listicle_diversity_topic_overlap_threshold: float = 0.3  # Skip penalty when topic overlap >= this

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

    # US-111-006: Enhanced video description keyword extraction
    # Max keywords to extract from video descriptions for embedding enrichment
    max_keywords_from_description: int = 5  # Includes both unigrams and n-grams
    ngram_enabled: bool = True  # Enable bigram/trigram extraction

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

    # US-95-005: Include video metadata in LLM reranker context
    reranker_include_metadata: bool = True  # Pass title, description, tags, chapters to LLM

    # US-134-007: Include transcript context in LLM reranker context
    # When enabled, extracts transcript snippets near the matched segment timestamp
    transcript_context_enabled: bool = True  # Pass transcript context to LLM
    transcript_context_chars: int = 200  # Max characters of transcript context to include

    # US-111-007: Context priority weights for metadata signals in LLM prompts
    # Controls how much weight the LLM gives to different context signals
    # when evaluating video candidates. Values should sum to 1.0.
    # Validated in ContextPriorityWeights.__post_init__
    context_priority_weights: Optional[ContextPriorityWeights] = None

    # US-95-010: Context richness calibration for confidence scores
    # When enabled, calibrates confidence based on available metadata context:
    # - Rich metadata (title + description + tags + chapters) -> higher confidence
    # - Sparse metadata -> conservative (lower) confidence
    context_richness_calibration: bool = True  # Enable confidence calibration based on context richness
    context_richness_boost_max: float = 0.08  # Max boost when all context signals present
    context_richness_penalty_max: float = 0.05  # Max penalty when no context signals

    # US-111-011: Individual signal weights for context richness calibration
    # Weights for each metadata signal - must sum to 1.0 for proper normalization
    # Controls how much each signal contributes to the richness score
    context_richness_title_weight: float = 0.25  # Weight for title signal
    context_richness_description_weight: float = 0.25  # Weight for description signal
    context_richness_tags_weight: float = 0.25  # Weight for tags signal
    context_richness_chapters_weight: float = 0.25  # Weight for chapters signal

    # US-141-002: Adaptive description truncation for context matching
    # When enabled, dynamically adjusts description truncation length based on keyword density:
    # - Longer descriptions with more keywords get more characters (up to max)
    # - Shorter/sparse descriptions get fewer characters (down to min)
    adaptive_description_truncation: bool = True  # Enable adaptive truncation
    min_description_chars: int = 100  # Minimum characters to use for description
    max_description_chars: int = 500  # Maximum characters to use for description

    # US-111-010: Voiceover context calibration for confidence scores
    # When enabled, calibrates confidence based on voiceover context availability:
    # - Rich context (segments before AND after) -> higher confidence (more context to verify)
    # - Limited context (no adjacent segments) -> conservative (less context to verify)
    voiceover_context_calibration: bool = True  # Enable confidence calibration based on voiceover context
    voiceover_context_boost_max: float = 0.05  # Max boost when rich voiceover context
    voiceover_context_penalty_max: float = 0.03  # Max penalty when limited/no voiceover context
    voiceover_context_window: int = 2  # Number of segments before/after to check (US-134-005: expanded from 1)

    # US-134-005: Voiceover topic continuity - boost when adjacent segments share topic keywords
    voiceover_topic_continuity: bool = True  # Enable topic continuity scoring
    voiceover_topic_continuity_boost: float = 0.03  # Max boost when topic continuity detected

    # US-134-005: Voiceover segment density - higher confidence for dense voiceover regions
    voiceover_segment_density: bool = True  # Enable segment density signal
    voiceover_segment_density_boost: float = 0.02  # Max boost for dense regions
    voiceover_segment_density_threshold: int = 4  # Min segments for dense boost

    # US-134-011: Context cache TTL - time-to-live for video context cache
    # Avoids rebuilding video context from metadata (title, description, tags, chapters)
    # for the same video across multiple segment comparisons
    context_cache_ttl_seconds: float = 3600.0  # 1 hour default TTL

    # US-141-003: Semantic context similarity scoring
    # When enabled, computes embedding-based similarity between voiceover context
    # and video metadata (title + description) to improve matching precision
    semantic_context_enabled: bool = True  # Enable semantic context similarity scoring
    semantic_context_weight: float = 0.10  # Weight for semantic similarity in confidence (0.0-1.0)

    # US-141-008: Temporal context tracking for confidence adjustment
    # Tracks context quality trends over time within a project run
    # - Improving context quality trends -> boost confidence
    # - Degrading context quality trends -> penalize confidence
    temporal_context_tracking_enabled: bool = True  # Enable temporal context tracking
    temporal_context_window: int = 10  # Number of segments to look back for trend detection
    temporal_boost_max: float = 0.03  # Max boost when context quality is improving

    # US-141-009: Context-aware candidate pre-filtering
    # Pre-filters candidates before expensive embedding computation
    # Uses title/description/tags overlap to quickly filter irrelevant candidates
    context_prefilter_enabled: bool = True  # Enable context-based pre-filtering
    context_filter_threshold: float = 0.20  # Min overlap score to keep candidate (0.0-1.0)

    # US-141-005: Title semantic expansion for better matching
    # When enabled, uses LLM to generate semantically related terms from video title
    # combined with voiceover context for improved keyword matching
    title_expansion_enabled: bool = True  # Enable title semantic expansion
    title_expansion_model: str = "gemini-2.0-flash"  # Model to use for title expansion
    title_expansion_max_terms: int = 10  # Maximum semantically related terms to generate
    title_expansion_weight: float = 0.05  # Weight for expanded terms in keyword matching

    # US-141-011: Multi-signal context boost optimization
    # When enabled, uses adaptive weights based on signal quality instead of equal weights
    # Quality scoring: title (length + keyword richness), description (length + density),
    #                 tags (count + specificity), chapters (count + coverage)
    adaptive_signal_weights: bool = True  # Enable adaptive signal weights
    signal_quality_weight: float = 0.10  # Max boost when all signals are high quality (0.0-1.0)

    # Delta matching (only match new videos)
    delta_matching_enabled: bool = True  # Enable delta-aware matching
    force_rematch: bool = False  # Force rematch all videos (CLI override)
    rematch_improvement_threshold: float = 0.05  # Re-evaluate if new video > existing + this

    # Chapter/topic matching
    chapter_matching_enabled: bool = True  # Enable chapter-based topic filtering
    enforce_chapter_boundaries: bool = True  # US-95-004: Penalize cross-chapter matches
    cross_chapter_penalty: float = 0.05  # US-95-004: Penalty for matching video from different chapter
    prefer_chapter_aligned_segments: bool = True  # US-95-011: Prefer segments aligned with chapter boundaries
    chapter_alignment_boost: float = 0.05  # US-95-011: Boost for chapter-aligned segments

    # US-134-012: Chapter timestamp features
    chapter_timestamp_context: bool = True  # Include relative timestamp in LLM context (e.g., "2:30 into video")
    chapter_boundary_awareness: bool = True  # Boost when voiceover segment aligns with chapter start
    chapter_boundary_boost: float = 0.03  # US-134-012: Boost for chapter boundary alignment

    # US-135-002: Configurable cross-chapter relevance weights
    # Weights for compute_relevance_matrix() in chapter_detection/bridge.py
    # When embedding_fn is provided, score = keyword_weight * jaccard + embedding_weight * cosine
    # When embedding_fn is None, score = pure jaccard (keyword_weight effectively 1.0)
    cross_chapter_keyword_weight: float = 0.6  # Weight for Jaccard keyword similarity in relevance matrix
    cross_chapter_embedding_weight: float = 0.4  # Weight for embedding cosine similarity in relevance matrix

    # Weights for compute_chapter_alignment_scores() in chapter_detection/bridge.py
    # Combined score = keyword_weight * keyword_score + temporal_weight * temporal_score + confidence_weight * confidence
    cross_chapter_alignment_keyword_weight: float = 0.5  # Weight for keyword/topic similarity
    cross_chapter_alignment_temporal_weight: float = 0.3  # Weight for temporal alignment (segment overlap)
    cross_chapter_alignment_confidence_weight: float = 0.2  # Weight for voiceover chapter detection confidence

    # US-105-011: Fallback strategy when no chapters are detected
    # - 'global': Disable chapter boosts and use standard global matching (default)
    # - 'segment': Fall back to segment-level matching without chapter grouping
    no_chapter_fallback_strategy: str = 'global'

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

    # Thematic consistency scoring (US-XXX)
    # Ensures adjacent segments maintain thematic coherence
    thematic_consistency_enabled: bool = True  # Enable thematic consistency scoring
    thematic_consistency_window: int = 3  # Number of adjacent segments to check
    thematic_consistency_boost_max: float = 0.05  # Maximum boost value

    # Cross-signal validation (US-XXX)
    # Validates consistency across multiple matching signals
    cross_signal_validation_enabled: bool = True  # Enable cross-signal consistency check
    consistency_penalty_max: float = 0.05  # Maximum penalty when signals are inconsistent (0.0-0.1)

    # Temporal overlap scoring (from MatchingScoringConfig - flat for backward compat)
    temporal_overlap_weight: float = 0.3  # Weight for temporal overlap (0.0-1.0)
    minimum_overlap_threshold: float = 0.3  # Minimum overlap required (below this, no boost applied)

    # View count context scoring (from MatchingScoringConfig - flat for backward compat)
    view_count_context_weight: float = 0.02  # Weight for view count as context signal (0 = disabled)
    view_count_boost_threshold: int = 1000000  # View count threshold for boost (1M views)

    # Visual-text fusion scoring (US-141-010 - from MatchingScoringConfig)
    visual_text_fusion_enabled: bool = True  # Enable visual-text fusion scoring
    visual_text_weight: float = 0.20  # Weight for visual component in fusion (0.0-1.0)

    # Channel reputation and engagement scoring (US-111-005)
    # Boosts confidence for videos from high-quality channels (high subscribers/engagement)
    channel_reputation_enabled: bool = True  # Enable channel reputation scoring
    channel_reputation_boost: float = 0.02  # Max boost for high-reputation channels
    channel_reputation_threshold: int = 100000  # Subscriber threshold for "high reputation" (100K)

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

    # US-134-008: Enhanced semantic coherence config
    semantic_window_size: int = 3  # Number of segments before/after to consider
    semantic_coherence_min_threshold: float = 0.5  # Minimum average coherence for boost
    topic_drift_detection_enabled: bool = True  # Detect topic shifts within chapters

    # Explanation confidence validation
    # When enabled, cross-checks that LLM explanation keywords appear in actual
    # voiceover/video text. When less than 50% of keywords are verifiable,
    # confidence is downgraded by 0.1 to penalize "hallucinated" explanations.
    explanation_validation_enabled: bool = True  # Enable explanation keyword verification

    # Location-aware matching (for travel/location content)
    location_matching: Optional[LocationMatchingConfig] = None

    # Scoring thresholds (US-53-002)
    scoring: Optional[MatchingScoringConfig] = None

    # Context enrichment (US-70-004)
    context_enrichment: Optional[ContextEnrichmentConfig] = None

    # Chapter-level source grouping (US-70-011)
    chapter_grouping: Optional[ChapterGroupingConfig] = None

    # Listicle topic extraction (US-105-005)
    listicle_topic: Optional[ListicleTopicConfig] = None

    # Listicle boundary pre-filtering (US-135-006)
    listicle_boundary: Optional[ListicleBoundaryConfig] = None

    # Voiceover topic extraction (US-111-002)
    voiceover_topic: Optional[VoiceoverTopicConfig] = None

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
            'max_keywords_from_description',  # US-111-006: Description keyword extraction
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

        if self.scoring is None:
            self.scoring = MatchingScoringConfig()
        elif isinstance(self.scoring, dict):
            self.scoring = MatchingScoringConfig(**self.scoring)

        # Sync flat fields to nested scoring config for backward compatibility
        # This allows config.yaml to use flat keys that get stored in nested config
        self.scoring.temporal_overlap_weight = self.temporal_overlap_weight
        self.scoring.minimum_overlap_threshold = self.minimum_overlap_threshold
        self.scoring.view_count_context_weight = self.view_count_context_weight
        self.scoring.view_count_boost_threshold = self.view_count_boost_threshold
        self.scoring.visual_text_fusion_enabled = self.visual_text_fusion_enabled
        self.scoring.visual_text_weight = self.visual_text_weight

        if self.context_enrichment is None:
            self.context_enrichment = ContextEnrichmentConfig()
        elif isinstance(self.context_enrichment, dict):
            self.context_enrichment = ContextEnrichmentConfig(**self.context_enrichment)

        if self.chapter_grouping is None:
            self.chapter_grouping = ChapterGroupingConfig()
        elif isinstance(self.chapter_grouping, dict):
            self.chapter_grouping = ChapterGroupingConfig(**self.chapter_grouping)

        # Listicle topic extraction (US-105-005)
        if self.listicle_topic is None:
            self.listicle_topic = ListicleTopicConfig()
        elif isinstance(self.listicle_topic, dict):
            self.listicle_topic = ListicleTopicConfig(**self.listicle_topic)

        # Listicle boundary pre-filtering (US-135-006)
        if self.listicle_boundary is None:
            self.listicle_boundary = ListicleBoundaryConfig()
        elif isinstance(self.listicle_boundary, dict):
            self.listicle_boundary = ListicleBoundaryConfig(**self.listicle_boundary)

        # Voiceover topic extraction (US-111-002)
        if self.voiceover_topic is None:
            self.voiceover_topic = VoiceoverTopicConfig()
        elif isinstance(self.voiceover_topic, dict):
            self.voiceover_topic = VoiceoverTopicConfig(**self.voiceover_topic)

        # Context priority weights (US-111-007) - validated in ContextPriorityWeights.__post_init__
        if self.context_priority_weights is None:
            self.context_priority_weights = ContextPriorityWeights()
        elif isinstance(self.context_priority_weights, dict):
            self.context_priority_weights = ContextPriorityWeights(**self.context_priority_weights)

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
