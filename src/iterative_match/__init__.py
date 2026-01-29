"""Iterative match package - Multi-pass gap filling for matching.

This package provides utilities for the IterativeMatchStage:
- gap_analyzer.py: Gap pattern classification
- query_learning.py: Track and learn from successful queries

Usage:
    from src.iterative_match import GapAnalysis, analyze_gaps
    from src.iterative_match import QueryLearningDB, QueryResult
"""

from .gap_analyzer import GapAnalysis, GapSegment, LockedMatch, analyze_gaps
from .query_learning import QueryLearningDB, QueryResult, QueryPlan

__all__ = [
    # Gap analysis
    'GapAnalysis',
    'GapSegment',
    'LockedMatch',
    'analyze_gaps',

    # Query learning
    'QueryLearningDB',
    'QueryResult',
    'QueryPlan',
]
