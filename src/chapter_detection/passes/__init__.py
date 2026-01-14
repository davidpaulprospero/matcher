"""
Chapter Detection Passes

Multi-pass detection strategy:
- Pass 1 (initial): Detect chapters using topic-based strategy
- Pass 2 (refinement): Adjust boundaries using embedding similarity
- Pass 3 (validation): Cross-validate with independent LLM call
- Pass 4 (coverage): Ensure complete, non-overlapping coverage
"""

from .initial import run_initial_detection
from .refinement import run_boundary_refinement
from .validation import run_validation
from .coverage import run_coverage_resolution

__all__ = [
    'run_initial_detection',
    'run_boundary_refinement',
    'run_validation',
    'run_coverage_resolution',
]
