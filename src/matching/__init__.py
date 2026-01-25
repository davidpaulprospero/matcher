"""
Matching package - Voiceover to video matching.

100% backward compatible with src/matching.py monolithic file.

This package is organized into focused modules:
- tracking.py: Timeline variety and clip deduplication
- llm_providers.py: LLM provider implementations (Gemini, Claude, Ollama)
- scoring.py: Confidence adjustment functions
- location_matching.py: Geographic filtering
- strategies.py: Alternative matching strategies (V4-V10)
- tiered_matcher.py: Core TieredMatcher implementation
- main.py: match_all_segments() orchestration

Public API (maintain backward compatibility):
- match_all_segments() - Main matching function
- TieredMatcher - Primary matching class
- StrategyMatcher - Alternative strategies
- GeminiMatcher, ClaudeMatcher, LocalLLMMatcher - LLM providers
- TimelineVarietyTracker, GlobalClipTracker - Tracking classes
"""

# Phase 1: Tracking
from .tracking import TimelineVarietyTracker, GlobalClipTracker

# Phase 2: LLM Providers
from .llm_providers import LLMProvider, GeminiMatcher, ClaudeMatcher, LocalLLMMatcher

# Phase 3: Strategies
from .strategies import StrategyMatcher

# Phase 4: Scoring (functions, not classes)
from . import scoring

# Phase 4b: Quality Metrics
from .metrics import MatchQualityMetrics, calculate_match_quality_metrics, log_quality_summary, log_confidence_histogram

# Phase 5: Location Matching
from .location_matching import LocationMatcher

# Phase 6: Main matching orchestration
from .main import match_all_segments

# Phase 7: TieredMatcher (temporary import from tiered_matcher.py bridge)
from .tiered_matcher import TieredMatcher

__all__ = [
    # Main API
    'match_all_segments',
    'TieredMatcher',  # Added for backward compatibility

    # Tracking
    'TimelineVarietyTracker',
    'GlobalClipTracker',

    # LLM Providers
    'GeminiMatcher',
    'ClaudeMatcher',
    'LocalLLMMatcher',

    # Strategies
    'StrategyMatcher',

    # Location Matching
    'LocationMatcher',

    # Quality Metrics
    'MatchQualityMetrics',
    'calculate_match_quality_metrics',
    'log_quality_summary',
    'log_confidence_histogram',
]
