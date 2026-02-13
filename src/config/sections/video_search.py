"""Video search configuration: Search budget and video search settings.

This module contains SearchBudgetConfig for Ralph's interview phase awareness
and VideoSearchConfig for the video search stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = [
    'SearchBudgetConfig',
    'VideoSearchConfig',
]


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
