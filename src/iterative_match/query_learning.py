"""Query Learning Database for Iterative Matching.

Tracks which query strategies work best for different gap types.
Persists learnings to disk for cross-project improvement.

Features:
- Per-pattern strategy success rates
- Successful query template tracking
- Progressive refinement suggestions

Created during IterativeMatchStage implementation (Jan 2026).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QueryPlan:
    """A planned search query with metadata."""
    query: str
    strategy: str  # 'voiceover', 'similar_locked', 'entity', 'topic'
    gap_indices: List[int]  # Which gaps this query targets
    seed_video_id: str = ""  # For similar_locked strategy
    priority: int = 0  # Higher = run first

    # Results (filled after execution)
    videos_found: int = 0
    executed: bool = False


@dataclass
class QueryResult:
    """Result of executing a search query."""
    query: str
    strategy: str
    gap_indices: List[int]
    videos_found: int
    gaps_filled: int  # How many gaps improved confidence
    avg_confidence_improvement: float
    successful: bool = False
    chapter_type: str = ''  # intro, body, conclusion, listicle_item, unknown

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyStats:
    """Statistics for a query strategy."""
    total_queries: int = 0
    total_gaps_targeted: int = 0
    total_gaps_filled: int = 0
    total_videos_found: int = 0
    success_rate: float = 0.0

    def record(self, result: QueryResult):
        """Record a query result."""
        self.total_queries += 1
        self.total_gaps_targeted += len(result.gap_indices)
        self.total_gaps_filled += result.gaps_filled
        self.total_videos_found += result.videos_found
        if self.total_gaps_targeted > 0:
            self.success_rate = self.total_gaps_filled / self.total_gaps_targeted

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyStats':
        return cls(**data)


class QueryLearningDB:
    """
    Tracks which query strategies work best for different gap types.

    Persists to JSON file for cross-project learning.

    Structure:
    {
        "version": "1.0",
        "pattern_strategy_success": {
            "abstract_concept": {"voiceover": 0.3, "similar_locked": 0.7, ...},
            "proper_noun": {"entity": 0.8, "voiceover": 0.4, ...},
            ...
        },
        "template_success": {
            "person footage": 15,
            "abstract concept video": 3,
            ...
        },
        "strategy_stats": {
            "voiceover": {"total_queries": 100, ...},
            ...
        }
    }
    """

    VERSION = "1.1"

    # Valid chapter types for bucketed tracking
    CHAPTER_TYPES = ("intro", "body", "conclusion", "listicle_item")

    def __init__(self, db_path: str = ".cache/query_learning.json"):
        """
        Initialize learning database.

        Args:
            db_path: Path to JSON file for persistence
        """
        self.db_path = Path(db_path)

        # Pattern -> strategy -> success rate (0.0 to 1.0)
        self.pattern_strategy_success: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # Query template -> success count
        self.template_success: Dict[str, int] = defaultdict(int)

        # Strategy -> aggregate stats
        self.strategy_stats: Dict[str, StrategyStats] = defaultdict(StrategyStats)

        # Chapter type -> strategy -> success rate (0.0 to 1.0)
        self.chapter_strategy_success: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # US-94-008: Template -> failure count (for negative keyword detection)
        self.template_failure: Dict[str, int] = defaultdict(int)

        # Default negative keyword patterns to exclude from search
        self.default_negative_keywords: List[str] = [
            "tutorial",
            "review",
            "unboxing",
            "explainer",
            "vs comparison",
            "explained",
        ]

        # Load existing data
        self._load()

    def _load(self):
        """Load learning database from disk."""
        if not self.db_path.exists():
            logger.debug(f"No existing learning DB at {self.db_path}")
            return

        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load pattern success rates
            for pattern, strategies in data.get('pattern_strategy_success', {}).items():
                for strategy, rate in strategies.items():
                    self.pattern_strategy_success[pattern][strategy] = rate

            # Load template success counts
            self.template_success.update(data.get('template_success', {}))

            # Load strategy stats
            for strategy, stats_dict in data.get('strategy_stats', {}).items():
                self.strategy_stats[strategy] = StrategyStats.from_dict(stats_dict)

            # Load chapter-type strategy success rates
            for chapter_type, strategies in data.get('chapter_strategy_success', {}).items():
                for strategy, rate in strategies.items():
                    self.chapter_strategy_success[chapter_type][strategy] = rate

            # US-94-008: Load template failure counts
            self.template_failure.update(data.get('template_failure', {}))

            logger.info(f"Loaded query learning DB with {len(self.pattern_strategy_success)} patterns")

        except Exception as e:
            logger.warning(f"Failed to load learning DB: {e}")

    def save(self):
        """Persist learning database to disk."""
        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'version': self.VERSION,
                'pattern_strategy_success': {
                    pattern: dict(strategies)
                    for pattern, strategies in self.pattern_strategy_success.items()
                },
                'template_success': dict(self.template_success),
                'strategy_stats': {
                    strategy: stats.to_dict()
                    for strategy, stats in self.strategy_stats.items()
                },
                'chapter_strategy_success': {
                    chapter_type: dict(strategies)
                    for chapter_type, strategies in self.chapter_strategy_success.items()
                },
                'template_failure': dict(self.template_failure),  # US-94-008
            }

            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved query learning DB to {self.db_path}")

        except Exception as e:
            logger.warning(f"Failed to save learning DB: {e}")

    def get_best_strategy(self, gap_pattern: str) -> str:
        """
        Return the strategy with highest success rate for this pattern.

        Args:
            gap_pattern: Pattern type (e.g., 'abstract_concept', 'proper_noun')

        Returns:
            Strategy name with best success rate, or 'voiceover' as default
        """
        strategies = self.pattern_strategy_success.get(gap_pattern, {})
        if not strategies:
            return 'voiceover'  # Default fallback

        return max(strategies.items(), key=lambda x: x[1])[0]

    def get_best_strategy_for_chapter(self, gap_pattern: str, chapter_type: str) -> str:
        """
        Return best strategy considering both gap pattern and chapter type.

        Blends pattern-level and chapter-level success rates (60/40 weighting)
        to prefer strategies that work well for this chapter type.

        Args:
            gap_pattern: Pattern type (e.g., 'abstract_concept')
            chapter_type: Chapter type (intro, body, conclusion, listicle_item)

        Returns:
            Strategy name with best blended success rate, or 'voiceover' as default
        """
        pattern_strategies = self.pattern_strategy_success.get(gap_pattern, {})
        chapter_strategies = self.chapter_strategy_success.get(chapter_type, {})

        if not pattern_strategies and not chapter_strategies:
            return 'voiceover'

        # Collect all known strategies
        all_strategies = set(pattern_strategies.keys()) | set(chapter_strategies.keys())
        if not all_strategies:
            return 'voiceover'

        # Blend: 60% pattern, 40% chapter type
        best_strategy = 'voiceover'
        best_score = -1.0
        for strategy in all_strategies:
            p_rate = pattern_strategies.get(strategy, 0.0)
            c_rate = chapter_strategies.get(strategy, 0.0)
            blended = 0.6 * p_rate + 0.4 * c_rate
            if blended > best_score:
                best_score = blended
                best_strategy = strategy

        return best_strategy

    def get_strategy_ranking_for_chapter(self, gap_pattern: str, chapter_type: str) -> List[str]:
        """
        Get strategies ranked by blended success rate for pattern + chapter type.

        Args:
            gap_pattern: Pattern type
            chapter_type: Chapter type (intro, body, conclusion, listicle_item)

        Returns:
            List of strategy names ordered by blended success rate (best first)
        """
        return self.get_multi_factor_strategy_ranking(
            gap_pattern, chapter_type, confidence=0.5, confidence_weight=0.0
        )

    def get_multi_factor_strategy_ranking(
        self,
        gap_pattern: str,
        chapter_type: str,
        confidence: float,
        confidence_weight: float = 0.2
    ) -> List[str]:
        """
        Get strategies ranked by multi-factor success rate combining pattern, chapter, and confidence.

        Combines:
        - Pattern success rate (60% weight of base score)
        - Chapter type success rate (40% weight of base score)
        - Confidence factor: low confidence favors conservative strategies (more keyword exploration),
          high confidence favors aggressive strategies (similar_to_locked)

        Args:
            gap_pattern: Pattern type (e.g., 'abstract_concept', 'proper_noun')
            chapter_type: Chapter type (intro, body, conclusion, listicle_item)
            confidence: Current confidence score for the gap (0.0 to 1.0)
            confidence_weight: How much to weight confidence factor (0.0 to 1.0, default 0.2)

        Returns:
            List of strategy names ordered by multi-factor success rate (best first)
        """
        # Conservative strategies: favor keyword exploration (good for low-confidence)
        # Aggressive strategies: favor similar_to_locked (good for high-confidence)
        CONSERVATIVE_STRATEGIES = ['voiceover', 'topic', 'entity']
        AGGRESSIVE_STRATEGIES = ['similar_locked']

        # Get base scores from pattern and chapter
        pattern_strategies = self.pattern_strategy_success.get(gap_pattern, {})
        chapter_strategies = self.chapter_strategy_success.get(chapter_type, {})

        # Collect all known strategies
        all_strategies = set(pattern_strategies.keys()) | set(chapter_strategies.keys())

        # Add defaults if no data
        if not all_strategies:
            all_strategies = {'voiceover', 'similar_locked', 'entity', 'topic'}

        # Calculate confidence bias: low confidence -> favor conservative, high -> favor aggressive
        # confidence_bias ranges from -1 (favor conservative) to +1 (favor aggressive)
        confidence_bias = (confidence - 0.5) * 2  # Maps 0->-1, 0.5->0, 1->+1

        scored = []
        for strategy in all_strategies:
            # Base score: 60% pattern, 40% chapter type
            p_rate = pattern_strategies.get(strategy, 0.0)
            c_rate = chapter_strategies.get(strategy, 0.0)
            base_score = 0.6 * p_rate + 0.4 * c_rate

            # Confidence bias adjustment
            if strategy in CONSERVATIVE_STRATEGIES:
                # Conservative strategies get boosted when confidence is low
                conf_adjustment = -confidence_bias * confidence_weight
            elif strategy in AGGRESSIVE_STRATEGIES:
                # Aggressive strategies get boosted when confidence is high
                conf_adjustment = confidence_bias * confidence_weight
            else:
                conf_adjustment = 0.0

            # Final score
            final_score = base_score + conf_adjustment
            scored.append((strategy, final_score))

        # Sort by final score (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in scored]

    def get_preferred_strategies(self, chapter_type: str, top_n: int = 3) -> List[str]:
        """
        Return the best strategies for a chapter type, ranked by success rate.

        Uses chapter-type-level success data only (no pattern blending).
        Falls back to default strategy order if no data exists.

        Args:
            chapter_type: Chapter type (intro, body, conclusion, listicle_item)
            top_n: Maximum number of strategies to return

        Returns:
            List of up to top_n strategy names ordered by success rate (best first)
        """
        default_order = ['voiceover', 'similar_locked', 'entity', 'topic']
        chapter_strategies = self.chapter_strategy_success.get(chapter_type, {})

        if not chapter_strategies:
            return default_order[:top_n]

        sorted_strategies = sorted(
            chapter_strategies.items(), key=lambda x: x[1], reverse=True
        )
        return [s[0] for s in sorted_strategies[:top_n]]

    def get_strategy_ranking(self, gap_pattern: str) -> List[str]:
        """
        Get strategies ranked by success rate for a pattern.

        Args:
            gap_pattern: Pattern type

        Returns:
            List of strategy names ordered by success rate (best first)
        """
        strategies = self.pattern_strategy_success.get(gap_pattern, {})
        if not strategies:
            # Return default order
            return ['voiceover', 'similar_locked', 'entity', 'topic']

        sorted_strategies = sorted(strategies.items(), key=lambda x: x[1], reverse=True)
        return [s[0] for s in sorted_strategies]

    def record_result(self, result: QueryResult, gap_pattern: str, chapter_type: str = "body"):
        """
        Update learning DB with query outcome.

        Args:
            result: Query result with success metrics
            gap_pattern: Pattern type of the gaps targeted
            chapter_type: Chapter type of the gap (intro, body, conclusion, listicle_item)
        """
        strategy = result.strategy

        # Update pattern -> strategy success rate (exponential moving average)
        current_rate = self.pattern_strategy_success[gap_pattern][strategy]
        new_rate = 1.0 if result.gaps_filled > 0 else 0.0
        # EMA with alpha=0.3 (recent results weighted more heavily)
        updated_rate = 0.3 * new_rate + 0.7 * current_rate
        self.pattern_strategy_success[gap_pattern][strategy] = updated_rate

        # Update chapter type -> strategy success rate (EMA)
        if chapter_type in self.CHAPTER_TYPES:
            current_ch_rate = self.chapter_strategy_success[chapter_type][strategy]
            updated_ch_rate = 0.3 * new_rate + 0.7 * current_ch_rate
            self.chapter_strategy_success[chapter_type][strategy] = updated_ch_rate

        # Update strategy stats
        self.strategy_stats[strategy].record(result)

        # Track successful query templates
        if result.gaps_filled > 0:
            # Extract template (remove specific terms, keep structure)
            template = self._extract_template(result.query)
            self.template_success[template] += result.gaps_filled

    def _extract_template(self, query: str) -> str:
        """
        Extract a generalizable template from a query.

        Removes specific names/terms, keeps query structure.

        Args:
            query: Original search query

        Returns:
            Templated version of query
        """
        # Replace capitalized words (names) with placeholder
        import re
        template = re.sub(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', '[NAME]', query)

        # Replace numbers with placeholder
        template = re.sub(r'\b\d+\b', '[NUM]', template)

        # Normalize whitespace
        template = ' '.join(template.split())

        return template.lower()

    def get_refined_template(self, base_query: str, pass_num: int) -> Optional[str]:
        """
        Return a refined query based on learned successful templates.

        Args:
            base_query: Original query to refine
            pass_num: Current pass number (higher = more aggressive refinement)

        Returns:
            Refined query string, or None if no refinement suggested
        """
        if pass_num < 2:
            return None  # Don't refine on first pass

        # Find similar successful templates
        base_template = self._extract_template(base_query)
        words = set(base_template.split())

        best_match = None
        best_score = 0

        for template, success_count in self.template_success.items():
            if success_count < 3:  # Need minimum evidence
                continue

            template_words = set(template.split())
            overlap = len(words & template_words)
            if overlap > best_score:
                best_score = overlap
                best_match = template

        if best_match and best_score >= 2:
            # Apply template structure to base query
            # This is a simple heuristic - keep base nouns, add template modifiers
            base_words = base_query.split()
            if len(base_words) < 3:
                return f"{base_query} footage video"
            return None  # No clear improvement

        return None

    def get_synonym_suggestions(self, word: str) -> List[str]:
        """
        Get synonym suggestions based on successful queries.

        Args:
            word: Word to find synonyms for

        Returns:
            List of words that appear in successful queries with similar context
        """
        # Simple heuristic: find words that co-occur in successful templates
        word_lower = word.lower()
        cooccurring = defaultdict(int)

        for template, count in self.template_success.items():
            if count < 2:
                continue
            words = template.split()
            if word_lower in words:
                for w in words:
                    if w != word_lower and w not in {'[name]', '[num]', 'video', 'footage'}:
                        cooccurring[w] += count

        if not cooccurring:
            return []

        # Return top co-occurring words
        sorted_words = sorted(cooccurring.items(), key=lambda x: x[1], reverse=True)
        return [w[0] for w in sorted_words[:3]]

    def record_failure(self, query: str):
        """
        Record a failing query template to improve future searches.

        US-94-008: Track queries that consistently fail to fill gaps,
        so we can exclude their patterns via negative keywords.

        Args:
            query: The query that failed to produce useful results
        """
        template = self._extract_template(query)
        self.template_failure[template] += 1

    def get_negative_keywords_for_query(
        self,
        query: str,
        threshold: int = 2
    ) -> List[str]:
        """
        Get negative keywords based on failing query patterns.

        US-94-008: Detect if the query contains words that commonly appear
        in failed queries, and suggest negative keywords to exclude.

        Args:
            query: The search query
            threshold: Minimum failure count to consider a pattern as "failing"

        Returns:
            List of negative keywords to add to exclude irrelevant results
        """
        negative_keywords = []
        query_lower = query.lower()

        # Check each default negative keyword pattern
        for neg_pattern in self.default_negative_keywords:
            # If the query already contains this pattern, it's likely to fail
            if neg_pattern.lower() in query_lower:
                negative_keywords.append(neg_pattern)

        # Check for words that have high failure rates
        words = query.split()
        for word in words:
            cleaned = word.lower().strip('.,!?;:\'"')
            if len(cleaned) > 3 and self.template_failure.get(cleaned, 0) >= threshold:
                negative_keywords.append(f"-{cleaned}")

        # Also check for common failing bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}".lower()
            if self.template_failure.get(bigram, 0) >= threshold:
                negative_keywords.append(f"-{bigram}")

        return negative_keywords

    def inject_negative_keywords(
        self,
        query: str,
        negative_patterns: Optional[List[str]] = None,
        enable_learning: bool = True
    ) -> str:
        """
        Inject negative keywords into a query to exclude irrelevant results.

        US-94-008: Modify the query to exclude common irrelevant content types
        (tutorials, reviews, unboxing videos) that typically don't match well.

        Args:
            query: Original search query
            negative_patterns: Custom patterns to exclude (defaults to class defaults)
            enable_learning: Whether to use learned failure patterns

        Returns:
            Query with negative keywords appended
        """
        if negative_patterns is None:
            negative_patterns = self.default_negative_keywords

        # Get learned negative keywords if enabled
        learned_negatives = []
        if enable_learning:
            learned_negatives = self.get_negative_keywords_for_query(query)

        # Combine default and learned negatives (avoid duplicates)
        all_negatives = set(negative_patterns) | set(learned_negatives)

        if not all_negatives:
            return query

        # Build exclusion string
        exclusion_parts = [f"-{neg}" for neg in all_negatives]
        exclusion_string = " ".join(exclusion_parts)

        return f"{query} {exclusion_string}"

    def get_failing_patterns(self, min_failures: int = 3) -> List[tuple]:
        """
        Get the most common failing query patterns.

        US-94-008: Return patterns that have consistently failed across runs,
        useful for debugging and improving query generation.

        Args:
            min_failures: Minimum failure count to include

        Returns:
            List of (template, failure_count) tuples sorted by failure count
        """
        failures = [
            (template, count)
            for template, count in self.template_failure.items()
            if count >= min_failures
        ]
        failures.sort(key=lambda x: x[1], reverse=True)
        return failures

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for reporting."""
        total_queries = sum(s.total_queries for s in self.strategy_stats.values())
        total_filled = sum(s.total_gaps_filled for s in self.strategy_stats.values())

        return {
            'total_queries_recorded': total_queries,
            'total_gaps_filled': total_filled,
            'patterns_learned': len(self.pattern_strategy_success),
            'templates_discovered': len(self.template_success),
            'chapter_types_learned': len(self.chapter_strategy_success),
            'strategy_success_rates': {
                strategy: stats.success_rate
                for strategy, stats in self.strategy_stats.items()
            },
        }
