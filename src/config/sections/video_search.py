"""Video search configuration: Search budget and video search settings.

This module contains SearchBudgetConfig for Ralph's interview phase awareness
and VideoSearchConfig for the video search stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    'SearchBudgetConfig',
    'PerKeywordCircuitBreakerConfig',
    'QueryTemplateConfig',
    'QuotaFallbackLevelsConfig',
    'QueryCacheConfig',
    'VideoSearchConfig',
]


@dataclass
class QueryCacheConfig:
    """US-156-003: Configuration for YouTube API query cache behavior.

    Controls cache invalidation and auto-refresh settings.
    """
    auto_invalidate_on_error: bool = True  # Auto-invalidate cache when API returns stale/empty data
    auto_invalidate_threshold: int = 3  # Consecutive empty results before invalidation
    cache_ttl_seconds: int = 0  # US-158-012: TTL for cached search results in seconds (0 = use default ttl_days)


@dataclass
class PerKeywordCircuitBreakerConfig:
    """Configuration for per-keyword circuit breaker in video search.

    Controls pause behavior between keyword searches based on failure history.
    This helps avoid rate limiting from YouTube when searching many keywords.
    """
    enabled: bool = True  # Enable per-keyword circuit breaker tracking
    consecutive_failures_threshold: int = 3  # Failures per keyword before pause
    pause_seconds: float = 30.0  # Base pause duration per keyword
    max_pause_seconds: float = 120.0  # Maximum pause cap per keyword
    jitter_factor: float = 0.2  # Random jitter factor (0.0 to 1.0)

    # US-113-004: Enable circuit breaker state persistence across runs
    # When true, per-keyword circuit breaker state is saved to checkpoint
    # and restored on pipeline resume.
    persist_state: bool = True

    # US-143-008: Auto-recovery with gradual reintroduction
    # Enable gradual traffic increase after circuit closes (half-open state)
    enable_recovery: bool = True
    # Maximum requests allowed during recovery phase (per cycle)
    recovery_max_requests: int = 3
    # Base for exponential backoff during recovery attempts
    recovery_backoff_base: float = 2.0
    # Maximum backoff multiplier during recovery
    recovery_max_backoff: float = 8.0
    # Consecutive successes needed to consider recovery complete
    recovery_success_threshold: int = 2


@dataclass
class SearchBudgetConfig:
    """Search budget configuration for Ralph interview phase.

    These values are read by Ralph during the interview phase to display
    search budget status and help users understand result distribution.
    """
    max_total_results: int = 200  # Maximum total video results across all keywords
    results_per_keyword: int = 20  # Default results per keyword (before adjustment)


@dataclass
class QueryTemplateConfig:
    """Query template configuration for US-153-011.

    Defines templates for different content types and A/B testing variants.
    """
    # Content type templates - format strings with {keyword} placeholder
    tutorial_templates: List[str] = field(default_factory=lambda: [
        "{keyword} tutorial",
        "{keyword} how to",
        "{keyword} guide",
    ])

    review_templates: List[str] = field(default_factory=lambda: [
        "{keyword} review",
        "{keyword} vs",
        "{keyword} comparison",
    ])

    vlog_templates: List[str] = field(default_factory=lambda: [
        "{keyword} vlog",
        "{keyword} day in life",
        "{keyword} experience",
    ])

    # Generic/broad templates (for general content)
    generic_templates: List[str] = field(default_factory=lambda: [
        "{keyword}",
        "{keyword} video",
        "{keyword} 2024",
    ])

    # A/B test variant templates
    # Control: original simple approach
    # Treatment: enhanced with content-type specific templates
    ab_variants: Dict[str, List[str]] = field(default_factory=lambda: {
        "control": ["{keyword}"],
        "treatment": [
            "{keyword}",
            "{keyword} tutorial",
            "{keyword} review",
            "{keyword} explained",
        ],
    })


@dataclass
class QuotaFallbackLevelsConfig:
    """Configuration for quota fallback levels (US-153-012).

    Controls progressive fallback from YouTube API to yt-dlp when quota
    is exhausted, with intermediate reduced-scope API usage.
    """
    # Tier 1: Full API - use full feature set
    # Tier 2: Reduced API - skip metadata/engagement, fewer results
    # Tier 3: yt-dlp fallback

    # Tier 1 -> Tier 2 threshold: % of quota remaining
    # When quota falls below this %, switch to reduced API mode
    tier1_to_tier2_threshold: float = 30.0  # 30% remaining

    # Tier 2 -> Tier 3 threshold: % of quota remaining
    # When quota falls below this %, switch to yt-dlp
    tier2_to_tier3_threshold: float = 10.0  # 10% remaining

    # Tier 2 (reduced scope) settings
    tier2_max_results_reduction: float = 0.5  # Return 50% of requested results
    tier2_skip_channel_metadata: bool = True  # Skip channel metadata enrichment
    tier2_skip_engagement_metrics: bool = True  # Skip engagement metrics (views, likes, comments)

    # Tier 3 (yt-dlp fallback) settings
    tier3_enabled: bool = True  # Enable yt-dlp fallback
    tier3_max_results: int = 10  # Max results from yt-dlp fallback

    # Enable progressive fallback
    enabled: bool = True

    def __post_init__(self):
        """Validate and normalize values."""
        if self.tier1_to_tier2_threshold < 0:
            self.tier1_to_tier2_threshold = 0.0
        if self.tier1_to_tier2_threshold > 100:
            self.tier1_to_tier2_threshold = 100.0

        if self.tier2_to_tier3_threshold < 0:
            self.tier2_to_tier3_threshold = 0.0
        if self.tier2_to_tier3_threshold > 100:
            self.tier2_to_tier3_threshold = 100.0

        if self.tier2_to_tier3_threshold >= self.tier1_to_tier2_threshold:
            # Ensure tier2 threshold is lower than tier1
            self.tier2_to_tier3_threshold = self.tier1_to_tier2_threshold - 5.0


@dataclass
class VideoSearchConfig:
    """Video search stage configuration.

    Controls YouTube video search behavior, including result limits,
    filtering, deduplication, and channel diversity.
    """
    # Result limits
    results_per_keyword: int = 20
    max_total_results: int = 200
    results_per_page: int = 50  # Results per page (max 50 for YouTube search API)

    # Budget distribution
    search_budget_aware: bool = True
    auto_distribute_budget: bool = True

    # Timeouts
    search_timeout: int = 30

    # Filters
    apply_duration_filter: bool = True
    apply_title_blacklist: bool = True
    apply_llm_title_filter: bool = True
    deduplicate_across_keywords: bool = True

    # Channel diversity
    enable_channel_diversity: bool = True
    max_videos_per_channel: int = 3

    # US-149-005: Search result deduplication and freshness scoring
    enable_deduplication: bool = True  # Enable duplicate video detection by title similarity
    max_title_similarity: float = 0.85  # Max similarity (0-1) to consider titles as duplicates
    enable_freshness_scoring: bool = True  # Enable freshness scoring based on publish date
    min_freshness_days: int = 365  # Videos older than this get reduced freshness score (0 = disabled)

    # US-146-006: Channel quality filtering and scoring
    min_subscriber_threshold: int = 1000  # Minimum subscribers to include video (0 = disabled)
    enable_channel_quality_score: bool = True  # Boost ranking based on channel metrics
    channel_quality_boost_factor: float = 0.05  # Max boost for high-quality channels
    # US-153-006: Channel enrichment settings
    include_channel_metadata: bool = True  # Fetch and include channel metadata (subscribers, videos, verified)

    # Query expansion
    use_tags_in_search: bool = True
    use_description_context: bool = True
    use_negative_context: bool = False

    # US-153-011: Query template optimization
    query_expansion_enabled: bool = True  # Enable query template variations
    max_query_variations: int = 3  # Maximum number of query variations to generate
    enable_ab_testing: bool = False  # Enable A/B testing for query templates
    ab_test_variant: str = "control"  # A/B test variant: "control" or "treatment"

    # US-157-004: Query optimization settings
    max_query_length: int = 256  # Maximum query length in characters
    enable_stopword_removal: bool = True  # Remove common stopwords from queries
    enable_query_expansion: bool = True  # Expand queries with related terms
    enable_relevance_scoring: bool = True  # Score results by title/description relevance
    relevance_boost_factor: float = 0.1  # Max boost for highly relevant results
    min_relevance_score: float = 0.3  # Minimum relevance score to include (0-1)
    stopword_list: Optional[List[str]] = None  # Custom stopwords (uses default if None)

    # Query length optimization
    short_query_length: int = 3  # Max words for broad queries
    long_query_length: int = 8  # Min words for specific queries

    # US-98-005: Chapter-based query generation
    use_chapter_queries: bool = True  # Use chapter topics for targeted search

    # US-98-008: Listicle topic keywords as search terms
    listicle_topic_as_search_terms: bool = True  # Use listicle topics for targeted search

    # Negative keywords (list)
    negative_keywords: Optional[List[str]] = None

    # Topic tags (dict)
    topic_tags: Optional[Dict[str, List[str]]] = None

    # US-113-002: Per-keyword circuit breaker
    per_keyword_circuit_breaker: Optional[PerKeywordCircuitBreakerConfig] = None

    # US-153-011: Query template configuration
    query_template: Optional[QueryTemplateConfig] = None

    # US-153-012: Quota fallback levels for progressive degradation
    quota_fallback_levels: Optional[QuotaFallbackLevelsConfig] = None

    # US-156-003: Query cache configuration
    query_cache: Optional[QueryCacheConfig] = None

    # US-156-005: Query sanitization and deduplication
    deduplicate_searches: bool = True  # Enable deduplication of search queries

    # US-158-012: Cache TTL for search results in seconds
    cache_ttl_seconds: int = 3600  # Cache search results for 1 hour (default)

    # US-156-007: Partial failure handling for batch operations
    max_partial_failure_percent: float = 20.0  # Max % of failed videos before warning/logging

    def __post_init__(self):
        """Set defaults for nested fields."""
        if self.negative_keywords is None:
            self.negative_keywords = [
                "trailer", "teaser", "compilation", "best of", "top 10"
            ]
        # US-157-004: Default stopword list if not provided
        if self.stopword_list is None:
            self.stopword_list = [
                "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
                "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
                "be", "have", "has", "had", "do", "does", "did", "will", "would",
                "could", "should", "may", "might", "must", "can", "this", "that",
                "these", "those", "i", "you", "he", "she", "it", "we", "they",
                "my", "your", "his", "her", "its", "our", "their", "what", "which",
                "who", "whom", "whose", "where", "when", "why", "how", "all", "each",
                "every", "both", "few", "more", "most", "other", "some", "such",
                "no", "nor", "not", "only", "own", "same", "so", "than", "too",
                "very", "just", "also", "now", "here", "there", "then", "once",
            ]
        if self.topic_tags is None:
            self.topic_tags = {
                "nature": ["wildlife", "landscape", "outdoor"],
                "travel": ["adventure", "destination", "culture"],
                "technology": ["innovation", "science", "gadgets"],
                "cooking": ["recipe", "food", "kitchen"],
                "fitness": ["workout", "exercise", "health"],
                "music": ["concert", "performance", "artist"],
                "documentary": ["history", "science", "nature"],
                "tutorial": ["howto", "guide", "education"],
            }
        # US-113-002: Convert dict to PerKeywordCircuitBreakerConfig if needed
        if self.per_keyword_circuit_breaker is None or isinstance(self.per_keyword_circuit_breaker, dict):
            self.per_keyword_circuit_breaker = PerKeywordCircuitBreakerConfig(
                **(self.per_keyword_circuit_breaker or {})
            )
        # US-153-011: Convert dict to QueryTemplateConfig if needed
        if self.query_template is None or isinstance(self.query_template, dict):
            self.query_template = QueryTemplateConfig(
                **(self.query_template or {})
            )
        # US-153-012: Convert dict to QuotaFallbackLevelsConfig if needed
        if self.quota_fallback_levels is None or isinstance(self.quota_fallback_levels, dict):
            self.quota_fallback_levels = QuotaFallbackLevelsConfig(
                **(self.quota_fallback_levels or {})
            )
        # US-156-003: Convert dict to QueryCacheConfig if needed
        if self.query_cache is None or isinstance(self.query_cache, dict):
            self.query_cache = QueryCacheConfig(
                **(self.query_cache or {})
            )
