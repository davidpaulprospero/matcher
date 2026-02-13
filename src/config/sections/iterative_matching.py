"""Iterative Matching configuration: Multi-pass gap filling for 90%+ confidence.

Implements iterative search passes to fill matching gaps and enforce source spacing.

Created during IterativeMatchStage implementation (Jan 2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

__all__ = [
    'IterativeMatchingConfig',
]


@dataclass
class IterativeMatchingConfig:
    """Iterative matching settings for multi-pass gap filling.

    Chain-of-thought: After initial matching, many segments have low confidence
    Reasoning: Iterative search with smart strategies can fill gaps
    Decision: Multi-pass with source spacing rule (300s) for visual variety

    Core Rules:
    - Locked: Confidence >= target AND source not used within spacing window
    - Gap: Confidence < target OR source spacing violated
    - Source Spacing: Same YouTube video only once per 300 seconds (5 minutes)
    - Preservation: Locked matches carry forward unchanged

    Algorithm per pass:
    1. Identify gaps/locks based on confidence + source spacing
    2. If gap% < min_gap_percentage or gaps = 0: STOP
    3. Analyze gap patterns (abstract concepts, proper nouns, etc.)
    4. Generate queries using multi-strategy approach
    5. If pass > 1: Apply progressive refinement
    6. YouTube search + caption fetch for new candidates
    7. Re-match ONLY gap segments with expanded pool
    8. Track which strategies succeeded -> update learning DB
    9. Merge: locked (unchanged) + updated gaps
    10. Repeat until max_iterations
    """

    # Master enable/disable
    enabled: bool = True

    # Confidence and spacing thresholds
    target_confidence: float = 0.90  # Confidence threshold for locking
    source_spacing_seconds: float = 300.0  # 5 minutes between same source

    # Iteration limits
    max_iterations: int = 5  # Maximum passes to attempt
    min_gap_percentage: float = 0.05  # Stop if fewer than 5% gaps remaining
    no_progress_min_pass: int = 3  # Minimum pass number before stopping on zero progress

    # Search settings
    search_results_per_gap: int = 10  # YouTube results to fetch per gap query
    max_new_videos_per_pass: int = 50  # Cap on new videos per iteration

    # Multi-strategy search (which strategies to use)
    use_voiceover_text_queries: bool = True  # Extract keywords from voiceover text
    use_similar_to_locked: bool = True  # Find videos similar to successful matches
    use_entity_topic_queries: bool = True  # Query with entities and topics
    use_description_queries: bool = True  # US-70-012: Generate queries from matched video descriptions
    use_tag_queries: bool = True  # US-73-009: Inject video tags into gap-filling queries
    parallel_strategy_search: bool = True  # Run strategies in parallel

    # Progressive refinement (for subsequent passes)
    enable_progressive_refinement: bool = True  # Enable query refinement on retry
    refinement_synonyms: bool = True  # Add synonyms to failing queries
    refinement_broaden_scope: bool = True  # Remove specific terms on retry

    # Gap pattern analysis
    analyze_gap_patterns: bool = True  # Classify gaps for smarter search
    gap_pattern_categories: List[str] = field(default_factory=lambda: [
        "abstract_concept",  # Hard to visualize: freedom, love, justice
        "proper_noun",  # Names, places, brands
        "action_verb",  # Running, cooking, swimming
        "location",  # Geographic references
        "emotion",  # Sentiment-heavy content
    ])

    # US-94-005: Confidence-based gap categorization
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 0.3,    # Gaps below this need aggressive search
        "medium": 0.6,  # Gaps between low and medium
        "high": 1.0,   # Gaps above medium need minimal search
    })

    # US-94-007: Duration-based gap prioritization
    # Longer gaps (>30 seconds) get priority boost in search order
    duration_priority_weight: float = 0.1  # Weight for duration-based boost

    # US-94-008: Negative keyword injection
    # Exclude irrelevant content types (tutorial, review, unboxing) from search results
    enable_negative_keywords: bool = True  # Enable negative keyword injection
    negative_keyword_patterns: List[str] = field(default_factory=lambda: [
        "tutorial",      # How-to content
        "review",        # Product reviews
        "unboxing",      # Product unboxing
        "explainer",     # Explainer videos
        "vs comparison", # Comparison videos
        "explained",     # Explained content
    ])

    # US-94-010: Duration tier diversity enforcement
    # Prefer diverse duration tiers (short <2min, medium 2-10min, long >10min) across matches
    tier_diversity_weight: float = 0.15  # Weight for duration tier diversity bonus

    # US-94-011: Query result caching
    # Cache YouTube search results to avoid repeated API calls
    cache_query_results: bool = True  # Enable/disable query result caching
    query_cache_ttl_hours: int = 24  # TTL for cached query results (hours)

    # US-94-012: Voiceover context awareness for gap keywords
    # Use adjacent segment text to enrich gap keywords
    context_window_segments: int = 1  # Number of adjacent segments to include for context

    # Query learning (track what works)
    enable_query_learning: bool = True  # Learn from successful queries
    learning_db_path: str = ".cache/query_learning.json"  # Learning DB file

    # Caption settings for new videos
    fetch_captions_for_new_videos: bool = True  # Fetch YouTube captions
    caption_timeout: int = 30  # Timeout per caption fetch
    caption_batch_size: int = 10  # Fetch captions in batches to avoid overwhelming the pipeline
    caption_fetch_delay: float = 0.5  # Delay between caption fetches to avoid rate limiting (seconds)

    # US-99-008: Duration filter for iterative search videos
    # Minimum and maximum video duration for gap-filling search results
    search_min_duration: int = 30  # Minimum duration in seconds
    search_max_duration: int = 600  # Maximum duration in seconds (10 minutes)

    # Logging and metrics
    log_pass_summaries: bool = True  # Log summary after each pass
    store_strategy_metrics: bool = True  # Track per-strategy effectiveness

    # US-89-006: Resume settings
    resume_budget_check: bool = True  # Check budget on resume and stop if exhausted
    max_queries_per_run: int = 100  # Maximum queries per run before stopping

    # US-101-006: Query budget tracking
    max_queries_per_pass: int = 20  # Maximum queries per pass
    budget_warning_threshold: float = 0.8  # Warning threshold (80% of budget)

    def __post_init__(self):
        """Validate configuration values."""
        # Clamp confidence to valid range
        self.target_confidence = max(0.0, min(1.0, self.target_confidence))

        # Ensure positive values
        self.source_spacing_seconds = max(0.0, self.source_spacing_seconds)
        self.max_iterations = max(1, self.max_iterations)
        self.no_progress_min_pass = max(1, self.no_progress_min_pass)
        self.min_gap_percentage = max(0.0, min(1.0, self.min_gap_percentage))
        self.search_results_per_gap = max(1, self.search_results_per_gap)
        self.max_new_videos_per_pass = max(1, self.max_new_videos_per_pass)
        self.caption_batch_size = max(1, self.caption_batch_size)
        self.caption_fetch_delay = max(0.0, self.caption_fetch_delay)
        self.query_cache_ttl_hours = max(0, self.query_cache_ttl_hours)
        self.context_window_segments = max(0, self.context_window_segments)  # US-94-012

        # US-99-008: Validate duration filters
        self.search_min_duration = max(0, self.search_min_duration)
        self.search_max_duration = max(self.search_min_duration, self.search_max_duration)  # Must be >= min

        # US-101-006: Validate query budget tracking
        self.max_queries_per_run = max(1, self.max_queries_per_run)
        self.max_queries_per_pass = max(1, self.max_queries_per_pass)
        self.budget_warning_threshold = max(0.0, min(1.0, self.budget_warning_threshold))
