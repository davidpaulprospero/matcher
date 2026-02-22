"""Iterative Matching configuration: Multi-pass gap filling for 90%+ confidence.

Implements iterative search passes to fill matching gaps and enforce source spacing.

Created during IterativeMatchStage implementation (Jan 2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

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

    # Query learning (track what works)
    enable_query_learning: bool = True  # Learn from successful queries
    learning_db_path: str = ".cache/query_learning.json"  # Learning DB file

    # Caption settings for new videos
    fetch_captions_for_new_videos: bool = True  # Fetch YouTube captions
    caption_timeout: int = 30  # Timeout per caption fetch
    caption_batch_size: int = 10  # Fetch captions in batches to avoid overwhelming the pipeline
    caption_fetch_delay: float = 0.5  # Delay between caption fetches to avoid rate limiting (seconds)

    # Logging and metrics
    log_pass_summaries: bool = True  # Log summary after each pass
    store_strategy_metrics: bool = True  # Track per-strategy effectiveness

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
