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
    'VideoSearchConfig',
]


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
class VideoSearchConfig:
    """Video search stage configuration.

    Controls YouTube video search behavior, including result limits,
    filtering, deduplication, and channel diversity.
    """
    # Result limits
    results_per_keyword: int = 20
    max_total_results: int = 200

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

    # Query expansion
    use_tags_in_search: bool = True
    use_description_context: bool = True
    use_negative_context: bool = False

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

    def __post_init__(self):
        """Set defaults for nested fields."""
        if self.negative_keywords is None:
            self.negative_keywords = [
                "trailer", "teaser", "compilation", "best of", "top 10"
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
