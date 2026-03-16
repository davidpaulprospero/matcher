"""Iterative match package - Multi-pass gap filling for matching.

This package provides utilities for the IterativeMatchStage:
- gap_analyzer.py: Gap pattern classification
- query_learning.py: Track and learn from successful queries

Usage:
    from src.iterative_match import GapAnalysis, analyze_gaps
    from src.iterative_match import QueryLearningDB, QueryResult
"""

from .gap_analyzer import (
    GapAnalysis,
    GapSegment,
    GapPatternLog,
    LockedMatch,
    analyze_gaps,
    analyze_gap_patterns_for_logging,
    log_gap_pattern_analysis,
)
from .query_learning import QueryLearningDB, QueryResult, QueryPlan

__all__ = [
    # Gap analysis
    'GapAnalysis',
    'GapSegment',
    'GapPatternLog',
    'LockedMatch',
    'analyze_gaps',
    'analyze_gap_patterns_for_logging',
    'log_gap_pattern_analysis',

    # Query learning
    'QueryLearningDB',
    'QueryResult',
    'QueryPlan',
]
