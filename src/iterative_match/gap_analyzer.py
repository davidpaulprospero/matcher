"""Gap Pattern Analyzer for Iterative Matching.

Classifies gap segments by pattern type to inform smarter search strategies.
Uses lightweight heuristics (POS tagging, word lists) rather than heavy ML models.

Gap patterns:
- abstract_concept: Hard to visualize (freedom, love, justice, hope)
- proper_noun: Named entities (people, places, brands, organizations)
- action_verb: Visual actions (running, cooking, flying)
- location: Geographic references (cities, countries, landmarks)
- emotion: Sentiment-heavy content (happy, sad, angry)

Created during IterativeMatchStage implementation (Jan 2026).
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from ..state import PipelineState, VoiceoverSegment

logger = logging.getLogger(__name__)


class ConfidenceCategory(Enum):
    """Confidence category for gap prioritization."""
    LOW = "low_confidence"      # < 0.3 - need aggressive search
    MEDIUM = "medium_confidence"  # 0.3-0.6 - moderate search effort
    HIGH = "high_confidence"    # > 0.6 - minimal search needed


# Word lists for pattern detection
ABSTRACT_CONCEPTS = {
    # Core abstract nouns
    'freedom', 'love', 'justice', 'hope', 'fear', 'peace', 'truth', 'beauty',
    'wisdom', 'courage', 'faith', 'trust', 'honor', 'pride', 'shame', 'guilt',
    'success', 'failure', 'happiness', 'sadness', 'anger', 'joy', 'sorrow',
    # Abstract qualities
    'integrity', 'loyalty', 'honesty', 'kindness', 'compassion', 'empathy',
    'ambition', 'determination', 'perseverance', 'resilience', 'patience',
    # Philosophical concepts
    'destiny', 'fate', 'purpose', 'meaning', 'existence', 'consciousness',
    'identity', 'reality', 'time', 'eternity', 'infinity', 'immortality',
    # Social concepts
    'democracy', 'equality', 'liberty', 'rights', 'power', 'authority',
    'community', 'society', 'culture', 'tradition', 'heritage', 'legacy',
}

EMOTION_WORDS = {
    # Positive emotions
    'happy', 'joyful', 'excited', 'thrilled', 'delighted', 'ecstatic',
    'content', 'satisfied', 'pleased', 'grateful', 'thankful', 'hopeful',
    'confident', 'proud', 'optimistic', 'enthusiastic', 'passionate',
    # Negative emotions
    'sad', 'unhappy', 'depressed', 'melancholy', 'gloomy', 'miserable',
    'angry', 'furious', 'enraged', 'irritated', 'frustrated', 'annoyed',
    'anxious', 'worried', 'nervous', 'stressed', 'overwhelmed', 'panicked',
    'scared', 'frightened', 'terrified', 'afraid', 'fearful', 'horrified',
    # Mixed/complex
    'confused', 'conflicted', 'ambivalent', 'nostalgic', 'bittersweet',
    'lonely', 'isolated', 'abandoned', 'rejected', 'humiliated', 'ashamed',
}

ACTION_VERBS = {
    # Physical actions (highly visual)
    'running', 'walking', 'jumping', 'flying', 'swimming', 'climbing',
    'dancing', 'fighting', 'driving', 'riding', 'sailing', 'surfing',
    # Manual activities
    'cooking', 'building', 'painting', 'drawing', 'writing', 'typing',
    'cutting', 'sewing', 'knitting', 'welding', 'hammering', 'drilling',
    # Sports
    'playing', 'kicking', 'throwing', 'catching', 'hitting', 'shooting',
    'scoring', 'winning', 'losing', 'racing', 'competing', 'training',
    # Work activities
    'working', 'teaching', 'learning', 'reading', 'studying', 'researching',
    'presenting', 'speaking', 'meeting', 'negotiating', 'planning', 'organizing',
}

LOCATION_INDICATORS = {
    # Prepositions
    'in', 'at', 'near', 'around', 'throughout', 'across', 'through',
    # Location words
    'city', 'town', 'village', 'country', 'nation', 'state', 'region',
    'continent', 'island', 'coast', 'beach', 'mountain', 'valley', 'river',
    'lake', 'ocean', 'sea', 'forest', 'desert', 'jungle', 'arctic',
    # Urban features
    'street', 'avenue', 'road', 'highway', 'bridge', 'building', 'tower',
    'park', 'square', 'plaza', 'market', 'station', 'airport', 'harbor',
    # Landmarks
    'landmark', 'monument', 'statue', 'museum', 'temple', 'church', 'mosque',
    'palace', 'castle', 'fortress', 'ruins', 'memorial', 'cemetery',
}


@dataclass
class GapSegment:
    """A segment identified as a gap (needs better match)."""
    segment_index: int
    confidence: float
    voiceover_text: str
    position: float  # Start time in timeline
    duration: float = 0.0  # Segment duration in seconds (US-94-007)
    pattern_type: str = ""  # Detected pattern
    confidence_category: str = ""  # low_confidence, medium_confidence, high_confidence
    keywords: List[str] = field(default_factory=list)
    chapter_id: Optional[str] = None  # Containing chapter/listicle group
    chapter_type: str = "body"  # intro, body, conclusion, listicle_item
    priority_boost: float = 0.0  # Boost for intro/conclusion chapters
    # US-135-007: Listicle position tracking
    gap_listicle_position: str = "none"  # 'boundary', 'middle', 'none'
    listicle_item_label: str = ""  # e.g., "first", "second", "#1" if at boundary


@dataclass
class LockedMatch:
    """A segment with a locked (good) match."""
    segment_index: int
    video_id: str
    confidence: float
    position: float
    title: str = ""


@dataclass
class GapAnalysis:
    """Analysis of gap patterns for smarter search."""

    # Pattern counts
    pattern_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Clustered gaps by pattern type
    clustered_gaps: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))

    # Specific pattern lists (gap indices)
    abstract_concepts: List[int] = field(default_factory=list)
    proper_nouns: List[int] = field(default_factory=list)
    action_descriptions: List[int] = field(default_factory=list)
    locations: List[int] = field(default_factory=list)
    emotional_content: List[int] = field(default_factory=list)
    other: List[int] = field(default_factory=list)

    def get_dominant_pattern(self) -> str:
        """Return the most common pattern type."""
        if not self.pattern_counts:
            return "other"
        return max(self.pattern_counts.items(), key=lambda x: x[1])[0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pattern_counts': dict(self.pattern_counts),
            'clustered_gaps': {k: list(v) for k, v in self.clustered_gaps.items()},
            'abstract_concepts': self.abstract_concepts,
            'proper_nouns': self.proper_nouns,
            'action_descriptions': self.action_descriptions,
            'locations': self.locations,
            'emotional_content': self.emotional_content,
            'other': self.other,
        }


# Default confidence thresholds
DEFAULT_CONFIDENCE_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 1.0,
}


def categorize_gaps_by_confidence(
    gaps: List[GapSegment],
    thresholds: Optional[Dict[str, float]] = None,
) -> List[GapSegment]:
    """
    Categorize gaps by confidence level for strategy selection.

    Categories:
    - low_confidence: confidence < 0.3 (or thresholds['low']) - needs aggressive search
    - medium_confidence: confidence 0.3-0.6 (or thresholds['medium']) - moderate search
    - high_confidence: confidence > 0.6 (or thresholds['high']) - minimal search needed

    Different strategies can be applied based on category:
    - Low: Use broader queries, more iterations, multiple strategies
    - Medium: Standard search with key term extraction
    - High: Quick confirmation search, rely on existing matches

    Args:
        gaps: List of GapSegment objects to categorize
        thresholds: Optional custom thresholds (dict with 'low', 'medium', 'high' keys)

    Returns:
        Gaps with confidence_category field populated
    """
    if not gaps:
        return gaps

    thresholds = thresholds or DEFAULT_CONFIDENCE_THRESHOLDS
    low_threshold = thresholds.get("low", 0.3)
    medium_threshold = thresholds.get("medium", 0.6)

    for gap in gaps:
        if gap.confidence < low_threshold:
            gap.confidence_category = ConfidenceCategory.LOW.value
        elif gap.confidence < medium_threshold:
            gap.confidence_category = ConfidenceCategory.MEDIUM.value
        else:
            gap.confidence_category = ConfidenceCategory.HIGH.value

    return gaps


def get_strategy_for_confidence_category(
    category: str,
) -> Dict[str, Any]:
    """
    Get search strategy parameters based on confidence category.

    Returns recommended strategy settings for each category:
    - low_confidence: Aggressive search with more results, broader queries
    - medium_confidence: Standard search approach
    - high_confidence: Minimal search, quick confirmation

    Args:
        category: The confidence category (low_confidence, medium_confidence, high_confidence)

    Returns:
        Dict with strategy parameters:
        - search_results: Number of results to fetch
        - use_broad_queries: Whether to use broader queries
        - max_iterations: Suggested max iterations for this gap
        - parallel_strategies: Whether to run strategies in parallel
    """
    strategies = {
        "low_confidence": {
            "search_results": 15,  # More results to find good match
            "use_broad_queries": True,
            "max_iterations": 5,  # More iterations allowed
            "parallel_strategies": True,
            "refine_on_fail": True,  # Always refine failed queries
        },
        "medium_confidence": {
            "search_results": 10,  # Standard results
            "use_broad_queries": False,
            "max_iterations": 3,
            "parallel_strategies": True,
            "refine_on_fail": True,
        },
        "high_confidence": {
            "search_results": 5,  # Quick confirmation only
            "use_broad_queries": False,
            "max_iterations": 1,
            "parallel_strategies": False,  # Single strategy is enough
            "refine_on_fail": False,
        },
    }

    return strategies.get(category, strategies["medium_confidence"])


def analyze_gaps(
    gaps: List[GapSegment],
    state: Optional['PipelineState'] = None,
    extracted_entities: Optional[List[Dict[str, Any]]] = None,
) -> GapAnalysis:
    """
    Classify gaps by pattern type to inform search strategy.

    Uses lightweight heuristics:
    - Word list matching for abstract concepts and emotions
    - Capitalization patterns for proper nouns
    - Verb detection for actions
    - Location indicator words

    Args:
        gaps: List of GapSegment objects to analyze
        state: Optional pipeline state for entity extraction
        extracted_entities: Optional pre-extracted entities list

    Returns:
        GapAnalysis with pattern classifications
    """
    analysis = GapAnalysis()

    # Get entities from state if not provided
    entities = extracted_entities or []
    if state and not entities:
        entities = getattr(state, 'extracted_entities', [])

    # Build entity name set for matching
    entity_names = set()
    for entity in entities:
        name = entity.get('name', '') if isinstance(entity, dict) else str(entity)
        if name:
            entity_names.add(name.lower())
            # Also add individual words from multi-word names
            entity_names.update(name.lower().split())

    for gap in gaps:
        pattern = _classify_gap_pattern(gap.voiceover_text, entity_names)
        gap.pattern_type = pattern

        # Update analysis
        analysis.pattern_counts[pattern] += 1
        analysis.clustered_gaps[pattern].append(gap.segment_index)

        # Update specific lists
        if pattern == 'abstract_concept':
            analysis.abstract_concepts.append(gap.segment_index)
        elif pattern == 'proper_noun':
            analysis.proper_nouns.append(gap.segment_index)
        elif pattern == 'action_verb':
            analysis.action_descriptions.append(gap.segment_index)
        elif pattern == 'location':
            analysis.locations.append(gap.segment_index)
        elif pattern == 'emotion':
            analysis.emotional_content.append(gap.segment_index)
        else:
            analysis.other.append(gap.segment_index)

    return analysis


# Default priority boosts for chapter positions
INTRO_PRIORITY_BOOST = 0.2
CONCLUSION_PRIORITY_BOOST = 0.15
DURATION_PRIORITY_THRESHOLD = 30.0  # Seconds - gaps longer than this get priority boost (US-94-007)
DEFAULT_DURATION_PRIORITY_WEIGHT = 0.1  # Default weight for duration-based priority


def annotate_gaps_with_chapters(
    gaps: List[GapSegment],
    total_segments: int,
    chapters: Optional[List[Dict[str, Any]]] = None,
    listicle_groups: Optional[List[Dict[str, Any]]] = None,
    intro_boost: float = INTRO_PRIORITY_BOOST,
    conclusion_boost: float = CONCLUSION_PRIORITY_BOOST,
    duration_priority_weight: float = DEFAULT_DURATION_PRIORITY_WEIGHT,
) -> List[GapSegment]:
    """
    Annotate gaps with chapter/listicle group info and apply priority boosts.

    Gaps in introduction chapters (first 10% of segments) get a priority boost
    of +0.2 (configurable). Gaps in conclusion chapters (last 10% of segments)
    get +0.15 (configurable). Longer gaps (>30 seconds) get additional boost
    proportional to their duration times the duration_priority_weight (US-94-007).

    Returns gaps sorted by effective priority (lower = higher priority):
    - priority = confidence - chapter_boost - duration_boost
    - duration_boost = (duration - DURATION_PRIORITY_THRESHOLD) * duration_priority_weight
      only applies when duration > DURATION_PRIORITY_THRESHOLD (30 seconds)

    Args:
        gaps: List of GapSegment objects to annotate.
        total_segments: Total number of voiceover segments (for position %).
        chapters: Optional chapter info from detection, each with
            'title', 'start_segment', 'end_segment' keys.
        listicle_groups: Optional listicle group info, each with
            'group_id', 'start_segment', 'end_segment' keys.
        intro_boost: Priority boost for intro chapter gaps.
        conclusion_boost: Priority boost for conclusion chapter gaps.
        duration_priority_weight: Weight for duration-based priority boost.

    Returns:
        Gaps sorted by priority (highest priority first).
    """
    if not gaps or total_segments <= 0:
        return gaps

    intro_threshold = total_segments * 0.10
    conclusion_threshold = total_segments * 0.90

    for gap in gaps:
        seg_idx = gap.segment_index

        # Annotate chapter_id from chapters or listicle_groups
        if chapters:
            for ch in chapters:
                ch_start = ch.get('start_segment', 0)
                ch_end = ch.get('end_segment', total_segments)
                if ch_start <= seg_idx < ch_end:
                    gap.chapter_id = ch.get('title', f"chapter_{ch_start}")
                    break

        if gap.chapter_id is None and listicle_groups:
            for grp in listicle_groups:
                grp_start = grp.get('start_segment', 0)
                grp_end = grp.get('end_segment', total_segments)
                if grp_start <= seg_idx < grp_end:
                    gap.chapter_id = grp.get('group_id', f"group_{grp_start}")
                    break

        # Classify chapter_type using position-based heuristics
        # Listicle items detected from listicle_groups take precedence
        if listicle_groups:
            for grp in listicle_groups:
                grp_start = grp.get('start_segment', 0)
                grp_end = grp.get('end_segment', total_segments)
                if grp_start <= seg_idx < grp_end:
                    gap.chapter_type = "listicle_item"
                    break

        if gap.chapter_type != "listicle_item":
            if seg_idx < intro_threshold:
                gap.chapter_type = "intro"
            elif seg_idx >= conclusion_threshold:
                gap.chapter_type = "conclusion"
            else:
                gap.chapter_type = "body"

        # Apply priority boost based on segment position
        if seg_idx < intro_threshold:
            gap.priority_boost = intro_boost
        elif seg_idx >= conclusion_threshold:
            gap.priority_boost = conclusion_boost

    # US-94-007: Apply duration-based priority boost
    # Longer gaps (>30 seconds) get additional boost
    def calculate_effective_priority(gap: GapSegment) -> float:
        """Calculate effective priority: lower = higher priority."""
        # Base: confidence minus chapter-based boost
        priority = gap.confidence - gap.priority_boost

        # Add duration-based boost for gaps >30 seconds
        if gap.duration > DURATION_PRIORITY_THRESHOLD:
            duration_boost = (gap.duration - DURATION_PRIORITY_THRESHOLD) * duration_priority_weight
            priority -= duration_boost  # Subtract to increase priority (lower = higher)

        return priority

    # Sort by effective priority: lower = higher priority
    gaps.sort(key=calculate_effective_priority)

    return gaps


# ============================================================================
# US-135-007: Listicle-aware gap analysis
# ============================================================================

# Position thresholds for determining if gap is at boundary or middle
# Boundary: within first/last 20% of a listicle group segments
LISTICLE_BOUNDARY_THRESHOLD = 0.20


def annotate_gaps_with_listicle_position(
    gaps: List[GapSegment],
    listicle_groups: Optional[List[Dict[str, Any]]] = None,
    boundary_threshold: float = LISTICLE_BOUNDARY_THRESHOLD,
) -> List[GapSegment]:
    """
    Annotate gaps with listicle position information.

    Determines if each gap is at a listicle group boundary (start or end),
    in the middle of a listicle group, or not in a listicle at all.

    Gaps at boundary trigger targeted queries (listicle item label + topic).
    Gaps in middle trigger broader queries (entire listicle theme).

    Args:
        gaps: List of GapSegment objects to annotate.
        listicle_groups: Optional listicle group info, each with
            'group_id', 'start_segment', 'end_segment', 'label' keys.
            Label is the listicle item label (e.g., "first", "#1", "1.").
        boundary_threshold: Fraction of group size for boundary detection.
            Default 0.20 = first/last 20% of group segments are boundaries.

    Returns:
        Gaps with gap_listicle_position and listicle_item_label populated.
    """
    if not gaps:
        return gaps

    if not listicle_groups:
        # Mark all gaps as 'none' if no listicle groups
        for gap in gaps:
            gap.gap_listicle_position = "none"
            gap.listicle_item_label = ""
        return gaps

    for gap in gaps:
        seg_idx = gap.segment_index
        gap.gap_listicle_position = "none"
        gap.listicle_item_label = ""

        # Find which listicle group contains this gap
        for grp in listicle_groups:
            grp_start = grp.get('start_segment', 0)
            grp_end = grp.get('end_segment', grp_start + 1)
            grp_size = grp_end - grp_start

            if grp_start <= seg_idx < grp_end:
                # Gap is in this listicle group
                # Determine position: boundary or middle
                position_in_group = seg_idx - grp_start
                boundary_size = max(1, int(grp_size * boundary_threshold))

                if position_in_group < boundary_size:
                    # At the start of the listicle group (boundary)
                    gap.gap_listicle_position = "boundary"
                elif position_in_group >= grp_size - boundary_size:
                    # At the end of the listicle group (boundary)
                    gap.gap_listicle_position = "boundary"
                else:
                    # In the middle of the listicle group
                    gap.gap_listicle_position = "middle"

                # Store the listicle item label if available
                label = grp.get('label', '')
                if label:
                    gap.listicle_item_label = label
                break

    return gaps


def get_query_strategy_for_listicle_position(
    gap: GapSegment,
    listicle_theme: str = "",
) -> Dict[str, Any]:
    """
    Get search query strategy based on gap's listicle position.

    Different strategies for boundary vs middle gaps:
    - Boundary: Use targeted query with listicle item label + topic keywords
    - Middle: Use broader query with entire listicle theme to find missed items
    - None: Use standard gap-based query strategy

    Args:
        gap: The gap segment to get strategy for.
        listicle_theme: Optional theme/topic of the entire listicle
            (useful for middle gaps to search more broadly).

    Returns:
        Dict with strategy parameters:
        - query_type: 'targeted', 'broad', or 'standard'
        - include_label: Whether to include listicle label in query
        - search_modifier: Additional search terms to add
        - description: Human-readable strategy description
    """
    position = gap.gap_listicle_position
    label = gap.listicle_item_label

    if position == "boundary":
        # Boundary: targeted search with item label
        return {
            "query_type": "targeted",
            "include_label": True,
            "search_modifier": label if label else "",
            "description": f"Targeted search with listicle label '{label}' + topic keywords"
        }
    elif position == "middle":
        # Middle: broader search with listicle theme
        return {
            "query_type": "broad",
            "include_label": False,
            "search_modifier": listicle_theme,
            "description": f"Broad search with listicle theme '{listicle_theme}' to find missed items"
        }
    else:
        # No listicle context - use standard strategy
        return {
            "query_type": "standard",
            "include_label": False,
            "search_modifier": "",
            "description": "Standard gap-based query (no listicle context)"
        }


def generate_listicle_aware_queries(
    gap: GapSegment,
    gap_keywords: List[str],
    listicle_theme: str = "",
    max_queries: int = 3,
) -> List[str]:
    """
    Generate search queries based on gap's listicle position.

    US-135-007: Uses listicle structure to generate more effective queries.
    - Boundary gaps: Combine listicle label with topic keywords
    - Middle gaps: Use broader listicle theme + topic keywords

    Args:
        gap: The gap segment to generate queries for.
        gap_keywords: Keywords extracted from gap voiceover text.
        listicle_theme: Theme/topic of the entire listicle (for middle gaps).
        max_queries: Maximum number of queries to generate.

    Returns:
        List of query strings optimized for listicle position.
    """
    if not gap_keywords:
        return []

    strategy = get_query_strategy_for_listicle_position(gap, listicle_theme)
    queries: List[str] = []

    if strategy["query_type"] == "targeted":
        # Boundary: targeted search with item label
        label = gap.listicle_item_label

        # Query 1: label + top keyword
        if label and gap_keywords:
            queries.append(f"{label} {gap_keywords[0]}")

        # Query 2: label + multiple keywords
        if label and len(gap_keywords) >= 2:
            queries.append(f"{label} {' '.join(gap_keywords[:2])}")

        # Query 3: just label for very targeted search
        if label:
            queries.append(label)

        # Fallback: keywords only if no label
        if not queries:
            queries.append(' '.join(gap_keywords[:3]))

    elif strategy["query_type"] == "broad":
        # Middle: broader search with listicle theme
        theme = listicle_theme.strip()

        # Query 1: theme + top keyword
        if theme and gap_keywords:
            queries.append(f"{theme} {gap_keywords[0]}")

        # Query 2: theme + multiple keywords
        if theme and len(gap_keywords) >= 2:
            queries.append(f"{theme} {' '.join(gap_keywords[:2])}")

        # Query 3: just theme for broad search
        if theme:
            queries.append(theme)

        # Fallback: keywords only
        if not queries:
            queries.append(' '.join(gap_keywords[:3]))

    else:
        # Standard: just gap keywords
        queries.append(' '.join(gap_keywords[:3]))
        if len(gap_keywords) > 3:
            queries.append(' '.join(gap_keywords[:2]))
        if len(gap_keywords) > 1:
            queries.append(gap_keywords[0])

    # Deduplicate while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        q_lower = q.lower().strip()
        if q_lower not in seen and q_lower:
            seen.add(q_lower)
            unique_queries.append(q)

    return unique_queries[:max_queries]


def _classify_gap_pattern(text: str, entity_names: Set[str]) -> str:
    """
    Classify a single text into a pattern category.

    Priority order (first match wins):
    1. Location (geographic context is specific)
    2. Proper noun (named entities)
    3. Action verb (visual activities)
    4. Abstract concept (hard to visualize)
    5. Emotion (sentiment content)
    6. Other (default)

    Args:
        text: Voiceover text to classify
        entity_names: Set of known entity names (lowercase)

    Returns:
        Pattern category string
    """
    text_lower = text.lower()
    words = set(text_lower.split())

    # 1. Check for location patterns
    if _has_location_pattern(text, words):
        return 'location'

    # 2. Check for proper nouns (capitalized words that aren't sentence starts)
    if _has_proper_noun(text, entity_names):
        return 'proper_noun'

    # 3. Check for action verbs
    action_overlap = words & ACTION_VERBS
    if len(action_overlap) >= 1:
        return 'action_verb'

    # 4. Check for abstract concepts
    abstract_overlap = words & ABSTRACT_CONCEPTS
    if len(abstract_overlap) >= 1:
        return 'abstract_concept'

    # 5. Check for emotional content
    emotion_overlap = words & EMOTION_WORDS
    if len(emotion_overlap) >= 1:
        return 'emotion'

    # 6. Default
    return 'other'


def _has_location_pattern(text: str, words: Set[str]) -> bool:
    """Check if text contains location references."""
    # Check for location indicator words
    location_overlap = words & LOCATION_INDICATORS
    if len(location_overlap) >= 2:  # Need multiple indicators
        return True

    # Check for "City, State" or "City, Country" patterns
    if re.search(r'[A-Z][a-z]+,\s*[A-Z][a-z]+', text):
        return True

    # Check for "in/at [Capitalized]" patterns
    if re.search(r'\b(in|at|near|from)\s+[A-Z][a-z]+\b', text):
        return True

    return False


def _has_proper_noun(text: str, entity_names: Set[str]) -> bool:
    """Check if text contains proper nouns."""
    text_lower = text.lower()

    # Check against known entities
    for entity in entity_names:
        if entity and entity in text_lower:
            return True

    # Find capitalized words that aren't at sentence start
    # Pattern: word boundary, space/punctuation, then Capital letter
    capitalized = re.findall(r'(?<=[.!?]\s)[A-Z][a-z]+|(?<=\s)[A-Z][a-z]+', text)

    # Filter out common sentence starters
    common_starters = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'i', 'we', 'you', 'it', 'he', 'she', 'they'}
    proper_nouns = [w for w in capitalized if w.lower() not in common_starters]

    return len(proper_nouns) >= 1


def extract_keywords_for_gap(
    gap: GapSegment,
    max_keywords: int = 5,
    context_text: str = ""
) -> List[str]:
    """
    Extract search-relevant keywords from a gap segment.

    US-94-012: Added voiceover context awareness - uses adjacent segment text
    to enrich gap keywords for better search relevance.

    Prioritizes:
    1. Proper nouns (capitalized words) from gap text
    2. Pattern-specific words (actions, locations)
    3. Content words (nouns, verbs, adjectives)
    4. Context words from adjacent segments (if available)

    Args:
        gap: GapSegment to extract keywords from
        max_keywords: Maximum keywords to return
        context_text: Optional text from adjacent segments for context enrichment.
            When provided, keywords from context are added with lower priority.

    Returns:
        List of keywords suitable for search queries
    """
    # Combine gap text with context for keyword extraction
    # Gap text takes precedence; context text provides supplementary keywords
    text = gap.voiceover_text
    context_lower = context_text.lower() if context_text else ""
    keywords = []

    # Stop words to exclude
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'us',
        'our', 'you', 'your', 'he', 'she', 'him', 'her', 'his', 'i', 'me', 'my',
    }

    # 1. Extract proper nouns (capitalized words)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    keywords.extend(proper_nouns[:2])

    # 2. Add pattern-specific keywords based on gap type
    words = text.lower().split()
    words = [w.strip('.,!?;:"\'-()[]') for w in words]
    words = [w for w in words if w and len(w) > 2 and w not in stop_words]

    if gap.pattern_type == 'action_verb':
        # Add action verbs found
        actions = [w for w in words if w in ACTION_VERBS]
        keywords.extend(actions[:2])
    elif gap.pattern_type == 'location':
        # Add location words
        locations = [w for w in words if w in LOCATION_INDICATORS]
        keywords.extend(locations[:2])
    elif gap.pattern_type == 'abstract_concept':
        # For abstract concepts, add concrete related words
        abstracts = [w for w in words if w in ABSTRACT_CONCEPTS]
        keywords.extend(abstracts[:1])

    # 3. Add remaining content words
    remaining = [w for w in words if w not in keywords and len(w) > 3]
    keywords.extend(remaining)

    # US-94-012: Add context keywords from adjacent segments
    # Context words are added with lower priority (after gap keywords)
    if context_lower:
        context_words = context_lower.split()
        context_words = [w.strip('.,!?;:"\'-()[]') for w in context_words]
        context_words = [w for w in context_words if w and len(w) > 3 and w not in stop_words]

        # Extract proper nouns from context (less prioritized)
        context_proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', context_text)
        for pn in context_proper_nouns:
            if pn.lower() not in [k.lower() for k in keywords]:
                context_words.insert(0, pn.lower())  # Prioritize proper nouns from context

        # Add unique context words that aren't already in keywords
        for cw in context_words:
            if cw.lower() not in [k.lower() for k in keywords] and len(keywords) < max_keywords:
                keywords.append(cw)

    # Deduplicate while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append(kw)

    return unique_keywords[:max_keywords]


@dataclass
class GapPatternLog:
    """Log entry for gap pattern analysis.

    Stores detailed analysis of gap patterns for logging and cross-run learning.
    """
    pass_number: int
    total_gaps: int
    pattern_summary: Dict[str, int] = field(default_factory=dict)

    # Topic clustering
    topic_clusters: Dict[str, List[int]] = field(default_factory=dict)  # common_topic -> gap indices
    topic_keywords: Dict[str, List[str]] = field(default_factory=dict)  # common_topic -> keywords

    # Timeline position clustering
    position_clusters: List[Dict[str, Any]] = field(default_factory=list)  # [{start, end, indices}]
    position_concentration: str = ""  # 'early', 'middle', 'late', 'spread'

    # Keyword patterns
    recurring_keywords: Dict[str, int] = field(default_factory=dict)  # keyword -> occurrence count
    keyword_patterns: List[str] = field(default_factory=list)  # Common keyword combinations

    # Query hints generated from patterns
    query_hints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pass_number': self.pass_number,
            'total_gaps': self.total_gaps,
            'pattern_summary': self.pattern_summary,
            'topic_clusters': self.topic_clusters,
            'topic_keywords': self.topic_keywords,
            'position_clusters': self.position_clusters,
            'position_concentration': self.position_concentration,
            'recurring_keywords': dict(self.recurring_keywords),
            'keyword_patterns': self.keyword_patterns,
            'query_hints': self.query_hints,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GapPatternLog':
        """Create from dictionary."""
        log = cls(
            pass_number=data.get('pass_number', 0),
            total_gaps=data.get('total_gaps', 0),
        )
        log.pattern_summary = data.get('pattern_summary', {})
        log.topic_clusters = data.get('topic_clusters', {})
        log.topic_keywords = data.get('topic_keywords', {})
        log.position_clusters = data.get('position_clusters', [])
        log.position_concentration = data.get('position_concentration', '')
        log.recurring_keywords = data.get('recurring_keywords', {})
        log.keyword_patterns = data.get('keyword_patterns', [])
        log.query_hints = data.get('query_hints', [])
        return log


def analyze_gap_patterns_for_logging(
    gaps: List[GapSegment],
    pass_number: int,
    total_duration: float = 0.0,
) -> GapPatternLog:
    """
    Analyze gaps for detailed pattern logging and query hint generation.

    This function identifies:
    - Gaps with similar voiceover topics (clustering by keywords)
    - Gaps at similar timeline positions (temporal clustering)
    - Recurring keyword patterns across gaps

    Args:
        gaps: List of GapSegment objects to analyze
        pass_number: Current iteration pass number
        total_duration: Total timeline duration for position analysis

    Returns:
        GapPatternLog with detailed analysis and query hints
    """
    log = GapPatternLog(pass_number=pass_number, total_gaps=len(gaps))

    if not gaps:
        return log

    # 1. Pattern summary (count by pattern type)
    for gap in gaps:
        pattern = gap.pattern_type or 'unclassified'
        log.pattern_summary[pattern] = log.pattern_summary.get(pattern, 0) + 1

    # 2. Topic clustering - group gaps by shared keywords
    log.topic_clusters, log.topic_keywords = _cluster_gaps_by_topic(gaps)

    # 3. Timeline position clustering
    log.position_clusters, log.position_concentration = _cluster_gaps_by_position(
        gaps, total_duration
    )

    # 4. Extract recurring keywords
    log.recurring_keywords = _extract_recurring_keywords(gaps)

    # 5. Identify keyword patterns (common combinations)
    log.keyword_patterns = _identify_keyword_patterns(gaps)

    # 6. Generate query hints based on all analyses
    log.query_hints = _generate_query_hints_from_patterns(log, gaps)

    return log


def _cluster_gaps_by_topic(
    gaps: List[GapSegment],
    min_cluster_size: int = 2,
) -> Tuple[Dict[str, List[int]], Dict[str, List[str]]]:
    """
    Cluster gaps by shared keywords/topics.

    Groups gaps that share significant content words.

    Args:
        gaps: Gaps to cluster
        min_cluster_size: Minimum gaps to form a cluster

    Returns:
        Tuple of (topic -> gap indices, topic -> keywords)
    """
    # Extract keywords for each gap
    gap_keywords: Dict[int, Set[str]] = {}
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
    }

    for gap in gaps:
        words = gap.voiceover_text.lower().split()
        words = [w.strip('.,!?;:"\'-()[]') for w in words]
        keywords = {w for w in words if len(w) > 3 and w not in stop_words}
        gap_keywords[gap.segment_index] = keywords

    # Find keyword overlap between gaps
    topic_clusters: Dict[str, List[int]] = defaultdict(list)
    topic_keywords: Dict[str, List[str]] = defaultdict(list)

    # Track which keywords appear in multiple gaps
    keyword_gaps: Dict[str, Set[int]] = defaultdict(set)
    for idx, keywords in gap_keywords.items():
        for kw in keywords:
            keyword_gaps[kw].add(idx)

    # Group by significant shared keywords
    processed_indices: Set[int] = set()
    cluster_id = 0

    for keyword, indices in sorted(keyword_gaps.items(), key=lambda x: -len(x[1])):
        if len(indices) < min_cluster_size:
            continue

        # Find gaps sharing this keyword that aren't already clustered
        new_cluster_indices = [i for i in indices if i not in processed_indices]
        if len(new_cluster_indices) < min_cluster_size:
            continue

        # Find common keywords among these gaps
        if new_cluster_indices:
            common_keywords = gap_keywords[new_cluster_indices[0]].copy()
            for idx in new_cluster_indices[1:]:
                common_keywords &= gap_keywords[idx]

            if common_keywords:
                topic_name = f"topic_{cluster_id}"
                topic_clusters[topic_name] = new_cluster_indices
                topic_keywords[topic_name] = list(common_keywords)[:5]  # Limit keywords
                processed_indices.update(new_cluster_indices)
                cluster_id += 1

    return dict(topic_clusters), dict(topic_keywords)


def _cluster_gaps_by_position(
    gaps: List[GapSegment],
    total_duration: float,
    window_seconds: float = 60.0,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Cluster gaps by timeline position.

    Identifies temporal patterns where gaps cluster together.

    Args:
        gaps: Gaps to analyze
        total_duration: Total timeline duration
        window_seconds: Time window for clustering

    Returns:
        Tuple of (position clusters, concentration description)
    """
    if not gaps:
        return [], 'none'

    # Sort gaps by position
    sorted_gaps = sorted(gaps, key=lambda g: g.position)
    positions = [g.position for g in sorted_gaps]

    # Find clusters using simple windowing
    clusters: List[Dict[str, Any]] = []
    current_cluster_indices: List[int] = []
    current_cluster_start = positions[0] if positions else 0

    for gap in sorted_gaps:
        if not current_cluster_indices:
            current_cluster_indices = [gap.segment_index]
            current_cluster_start = gap.position
        elif gap.position - current_cluster_start <= window_seconds:
            current_cluster_indices.append(gap.segment_index)
        else:
            # Save current cluster if significant
            if len(current_cluster_indices) >= 2:
                clusters.append({
                    'start': current_cluster_start,
                    'end': current_cluster_start + window_seconds,
                    'indices': current_cluster_indices,
                    'count': len(current_cluster_indices),
                })
            # Start new cluster
            current_cluster_indices = [gap.segment_index]
            current_cluster_start = gap.position

    # Don't forget last cluster
    if len(current_cluster_indices) >= 2:
        clusters.append({
            'start': current_cluster_start,
            'end': current_cluster_start + window_seconds,
            'indices': current_cluster_indices,
            'count': len(current_cluster_indices),
        })

    # Determine concentration
    if total_duration <= 0 or not positions:
        concentration = 'unknown'
    else:
        avg_position = sum(positions) / len(positions)
        relative_position = avg_position / total_duration

        if relative_position < 0.33:
            concentration = 'early'
        elif relative_position > 0.67:
            concentration = 'late'
        else:
            concentration = 'middle'

        # Check for spread distribution
        spread = (max(positions) - min(positions)) / total_duration if total_duration > 0 else 0
        if spread > 0.7:
            concentration = 'spread'

    return clusters, concentration


def _extract_recurring_keywords(
    gaps: List[GapSegment],
    min_occurrences: int = 2,
) -> Dict[str, int]:
    """
    Extract keywords that appear across multiple gaps.

    Args:
        gaps: Gaps to analyze
        min_occurrences: Minimum occurrences to be considered recurring

    Returns:
        Dict of keyword -> occurrence count
    """
    keyword_counts: Dict[str, int] = defaultdict(int)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
    }

    for gap in gaps:
        words = gap.voiceover_text.lower().split()
        words = [w.strip('.,!?;:"\'-()[]') for w in words]
        for word in words:
            if len(word) > 3 and word not in stop_words:
                keyword_counts[word] += 1

    # Filter to recurring keywords
    recurring = {k: v for k, v in keyword_counts.items() if v >= min_occurrences}
    # Sort by count descending
    return dict(sorted(recurring.items(), key=lambda x: -x[1]))


def _identify_keyword_patterns(
    gaps: List[GapSegment],
    max_patterns: int = 5,
) -> List[str]:
    """
    Identify common keyword combinations across gaps.

    Looks for bigrams and trigrams that appear frequently.

    Args:
        gaps: Gaps to analyze
        max_patterns: Maximum patterns to return

    Returns:
        List of common keyword pattern strings
    """
    bigram_counts: Dict[str, int] = defaultdict(int)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were',
    }

    for gap in gaps:
        words = gap.voiceover_text.lower().split()
        words = [w.strip('.,!?;:"\'-()[]') for w in words]
        content_words = [w for w in words if len(w) > 2 and w not in stop_words]

        # Count bigrams
        for i in range(len(content_words) - 1):
            bigram = f"{content_words[i]} {content_words[i+1]}"
            bigram_counts[bigram] += 1

    # Filter to patterns appearing multiple times
    patterns = [p for p, count in bigram_counts.items() if count >= 2]
    # Sort by frequency
    patterns.sort(key=lambda p: -bigram_counts[p])

    return patterns[:max_patterns]


def _generate_query_hints_from_patterns(
    log: GapPatternLog,
    gaps: List[GapSegment],
) -> List[str]:
    """
    Generate search query hints based on gap pattern analysis.

    Creates actionable suggestions for improving gap coverage.

    Args:
        log: Current GapPatternLog with analysis results
        gaps: Original gaps for additional context

    Returns:
        List of query hint strings
    """
    hints: List[str] = []

    # 1. Hints from recurring keywords
    top_keywords = list(log.recurring_keywords.keys())[:5]
    if top_keywords:
        hints.append(f"Consider searching for: {', '.join(top_keywords)}")

    # 2. Hints from topic clusters
    for topic, keywords in log.topic_keywords.items():
        cluster_size = len(log.topic_clusters.get(topic, []))
        if cluster_size >= 3 and keywords:
            hints.append(f"Cluster of {cluster_size} gaps share terms: {', '.join(keywords[:3])}")

    # 3. Hints from position concentration
    if log.position_concentration == 'early':
        hints.append("Gaps concentrated in early timeline - consider intro/setup footage")
    elif log.position_concentration == 'late':
        hints.append("Gaps concentrated in late timeline - consider conclusion/wrap-up footage")

    # 4. Hints from pattern types
    if log.pattern_summary:
        dominant = max(log.pattern_summary.items(), key=lambda x: x[1])
        pattern_type, count = dominant
        if count >= 3:
            if pattern_type == 'abstract_concept':
                hints.append(f"{count} gaps are abstract concepts - try metaphorical/symbolic footage")
            elif pattern_type == 'action_verb':
                hints.append(f"{count} gaps describe actions - search for action/activity footage")
            elif pattern_type == 'location':
                hints.append(f"{count} gaps reference locations - search for geographic/place footage")
            elif pattern_type == 'proper_noun':
                hints.append(f"{count} gaps mention specific names - search for named entity footage")
            elif pattern_type == 'emotion':
                hints.append(f"{count} gaps are emotional content - search for expressive/mood footage")

    # 5. Hints from keyword patterns
    if log.keyword_patterns:
        hints.append(f"Common word pairs: {', '.join(log.keyword_patterns[:3])}")

    return hints


def log_gap_pattern_analysis(
    log: GapPatternLog,
    logger_instance: Optional[logging.Logger] = None,
) -> None:
    """
    Log gap pattern analysis at DEBUG level.

    Writes detailed pattern information to the logger for optimization insights.

    Args:
        log: GapPatternLog to output
        logger_instance: Logger to use (defaults to module logger)
    """
    log_fn = logger_instance or logger

    log_fn.debug(f"=== Gap Pattern Analysis (Pass {log.pass_number}) ===")
    log_fn.debug(f"Total gaps: {log.total_gaps}")

    # Pattern summary
    if log.pattern_summary:
        pattern_str = ", ".join(f"{k}={v}" for k, v in log.pattern_summary.items())
        log_fn.debug(f"Pattern distribution: {pattern_str}")

    # Topic clusters
    if log.topic_clusters:
        log_fn.debug(f"Found {len(log.topic_clusters)} topic clusters:")
        for topic, indices in log.topic_clusters.items():
            keywords = log.topic_keywords.get(topic, [])
            log_fn.debug(f"  {topic}: {len(indices)} gaps, keywords={keywords}")

    # Position clustering
    if log.position_clusters:
        log_fn.debug(f"Found {len(log.position_clusters)} position clusters:")
        for cluster in log.position_clusters:
            log_fn.debug(
                f"  Time {cluster['start']:.1f}s-{cluster['end']:.1f}s: "
                f"{cluster['count']} gaps"
            )
    if log.position_concentration:
        log_fn.debug(f"Position concentration: {log.position_concentration}")

    # Recurring keywords
    if log.recurring_keywords:
        top_5 = list(log.recurring_keywords.items())[:5]
        keywords_str = ", ".join(f"{k}({v})" for k, v in top_5)
        log_fn.debug(f"Recurring keywords: {keywords_str}")

    # Query hints
    if log.query_hints:
        log_fn.debug("Query hints:")
        for hint in log.query_hints:
            log_fn.debug(f"  → {hint}")


# ============================================================================
# US-70-012: Description-derived search queries
# ============================================================================

# Stop words for description key phrase extraction (US-126-011: expanded)
_DESCRIPTION_STOP_WORDS = {
    # Standard English stopwords
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
    'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'us',
    'our', 'you', 'your', 'he', 'she', 'him', 'her', 'his', 'i', 'me', 'my',
    'not', 'no', 'so', 'if', 'then', 'than', 'very', 'just', 'about',
    'also', 'more', 'some', 'any', 'all', 'each', 'every', 'how', 'what',
    'when', 'where', 'which', 'who', 'why', 'here', 'there', 'only',
    # YouTube/website boilerplate
    'http', 'https', 'www', 'com', 'subscribe', 'like', 'video', 'channel',
    'facebook', 'twitter', 'instagram', 'tiktok', 'patreon', 'discord',
    'merch', 'store', 'shop', 'link', 'bio', 'follow', 'share', 'comment',
    # Common verbs (too generic for queries)
    'get', 'got', 'make', 'made', 'take', 'took', 'see', 'saw', 'know',
    'knew', 'think', 'thought', 'want', 'wanted', 'need', 'needed', 'use',
    'used', 'find', 'found', 'give', 'gave', 'tell', 'told', 'say', 'said',
    'go', 'went', 'come', 'came', 'look', 'looked', 'watch', 'watched',
    'play', 'played', 'learn', 'learned', 'check', 'checked', 'visit',
    # Common adjectives (too generic)
    'new', 'old', 'big', 'small', 'good', 'great', 'best', 'first',
    'last', 'next', 'other', 'many', 'much', 'most', 'same', 'different',
    'own', 'such', 'now', 'today', 'tomorrow', 'yesterday', 'day', 'time',
    'way', 'thing', 'things', 'people', 'year', 'years', 'day', 'days',
    # Prepositions and conjunctions
    'up', 'down', 'out', 'over', 'under', 'again', 'once', 'well', 'even',
    'still', 'back', 'still', 'while', 'being', 'because', 'after', 'before',
}

# Common proper noun patterns for NER-like extraction (US-126-011)
_PROPER_NOUN_PATTERNS = [
    # Acronyms and abbreviations (2-5 uppercase letters)
    r'\b([A-Z]{2,5})\b',
    # CamelCase words (e.g., "YouTube", "TikTok")
    r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b',
    # Names with titles
    r'\b(Dr|Mr|Mrs|Ms|Prof|Sir|Lord|King|Queen|President)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
    # Location patterns
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:City|Town|State|Country|Island|Mountain|River|Lake|Sea|Ocean|Park|Museum))\b',
    # Brand names (common patterns)
    r'\b([A-Z][a-z]+(?:\s+(?:Inc|Corp|LLC|Ltd|Co|Studio|Lab|Company|Brand)))\b',
]

# Compile proper noun regex patterns
_PROPER_NOUN_REGEXES = [re.compile(p) for p in _PROPER_NOUN_PATTERNS]


def derive_queries_from_descriptions(
    matched_videos: List[Dict[str, Any]],
    gap_segment: 'GapSegment',
    max_queries: int = 3,
) -> List[str]:
    """
    Derive search queries from descriptions of already-matched videos,
    targeted to a specific gap segment.

    Extracts top noun phrases from video descriptions using simple regex
    (capitalized word sequences, quoted phrases, proper nouns), then filters
    for relevance to the gap segment's voiceover text.

    US-126-011 Improvements:
    - Prefer noun phrases (2+ words) over single words
    - Add named entity recognition for proper nouns (locations, people, brands)
    - More aggressive stopword filtering

    Args:
        matched_videos: List of matched video dicts with 'description' field.
        gap_segment: The gap segment to derive queries for.
        max_queries: Maximum number of queries to return (default 3).

    Returns:
        List of query strings derived from descriptions.
    """
    if not matched_videos:
        return []

    # Collect descriptions from matched videos
    descriptions = []
    for video in matched_videos:
        desc = ''
        if isinstance(video, dict):
            desc = video.get('description', '') or ''
        else:
            desc = getattr(video, 'description', '') or ''
        if desc and len(desc.strip()) > 10:
            descriptions.append(desc)

    if not descriptions:
        return []

    # Extract noun phrases using regex: capitalized word sequences, quoted phrases, proper nouns
    noun_phrases: List[str] = []
    proper_nouns: List[str] = []  # US-126-011: Track proper nouns separately
    seen_lower: set = set()

    for desc in descriptions:
        # Remove URLs
        clean = re.sub(r'https?://\S+', '', desc)

        # 1. Quoted phrases (single or double quotes) - high quality phrases
        quoted = re.findall(r'["\u201c]([^"\u201d]{3,50})["\u201d]', clean)
        for phrase in quoted:
            phrase_stripped = phrase.strip()
            # US-126-011: Prefer phrases with 2+ words
            if phrase_stripped.lower() not in seen_lower and len(phrase_stripped.split()) >= 2:
                seen_lower.add(phrase_stripped.lower())
                noun_phrases.append(phrase_stripped)

        # 2. Capitalized word sequences (2-4 consecutive capitalized words)
        cap_sequences = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', clean)
        for seq in cap_sequences:
            # Filter out stopwords more aggressively (US-126-011)
            words = seq.lower().split()
            if all(w not in _DESCRIPTION_STOP_WORDS for w in words):
                if seq.lower() not in seen_lower:
                    seen_lower.add(seq.lower())
                    noun_phrases.append(seq)

        # 3. US-126-011: Extract proper nouns using patterns
        for regex in _PROPER_NOUN_REGEXES:
            matches = regex.findall(clean)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[-1]
                if match and len(match) > 2:
                    match_clean = match.strip()
                    if match_clean.lower() not in seen_lower:
                        seen_lower.add(match_clean.lower())
                        proper_nouns.append(match_clean)

    # If we have proper nouns, include them in the candidate list
    all_candidates = noun_phrases + proper_nouns

    if not all_candidates:
        # Fallback: use extract_description_queries for TF-IDF-like extraction
        return extract_description_queries(descriptions, max_queries=max_queries)

    # Score phrases by relevance to the gap segment's voiceover text
    gap_words = set(gap_segment.voiceover_text.lower().split())
    gap_words = {w.strip('.,!?;:"\'-()[]') for w in gap_words}
    gap_words = {w for w in gap_words if len(w) > 3}

    scored: List[tuple] = []
    for phrase in all_candidates:
        phrase_words = set(phrase.lower().split())
        # US-126-011: Prefer multi-word phrases over single words
        word_count = len(phrase_words)
        is_proper_noun = phrase in proper_nouns

        # Score: overlap with gap text + phrase length bonus + proper noun boost
        overlap = len(phrase_words & gap_words)
        base_score = overlap * 2.0

        # Prefer noun phrases (2+ words) over single words
        if word_count >= 2:
            base_score += word_count * 0.5  # Multi-word phrase bonus
        else:
            base_score -= 1.0  # Penalty for single words

        # Boost proper nouns slightly (they're often specific search terms)
        if is_proper_noun:
            base_score += 0.5

        scored.append((phrase, base_score))

    # Sort by score descending, take top N
    scored.sort(key=lambda x: -x[1])

    # US-126-011: Filter results - prefer multi-word phrases when available
    results: List[str] = []
    for phrase, score in scored:
        # If we have enough multi-word results, filter out single words
        if len(results) >= max_queries:
            break
        results.append(phrase)

    return results[:max_queries]


def extract_description_queries(
    descriptions: List[str],
    max_queries: int = 10,
    max_phrases_per_description: int = 3,
) -> List[str]:
    """
    Extract key phrases from video descriptions to generate search queries.

    Uses simple TF-IDF-like keyword extraction: finds words that appear frequently
    across descriptions and forms short phrases from them.

    Args:
        descriptions: List of video description strings from matched videos.
        max_queries: Maximum number of queries to return.
        max_phrases_per_description: Maximum phrases to extract per description.

    Returns:
        List of search query strings derived from descriptions.
    """
    if not descriptions:
        return []

    # Count word frequency across all descriptions (TF-IDF-like)
    word_freq: Dict[str, int] = defaultdict(int)
    word_in_docs: Dict[str, int] = defaultdict(int)  # Document frequency

    for desc in descriptions:
        if not desc:
            continue
        words = _extract_content_words(desc)
        unique_words = set(words)
        for word in words:
            word_freq[word] += 1
        for word in unique_words:
            word_in_docs[word] += 1

    if not word_freq:
        return []

    num_docs = len([d for d in descriptions if d])
    if num_docs == 0:
        return []

    # Score words: prefer frequent but not ubiquitous words
    word_scores: Dict[str, float] = {}
    for word, freq in word_freq.items():
        doc_freq = word_in_docs.get(word, 1)
        # Simple TF-IDF-like scoring: higher freq is good, but penalize
        # words that appear in every single description (too generic)
        if doc_freq >= num_docs and num_docs > 1:
            # Appears in all docs — likely too generic
            word_scores[word] = freq * 0.3
        else:
            word_scores[word] = freq * (1.0 + 1.0 / max(doc_freq, 1))

    # Get top-scoring words
    top_words = sorted(word_scores.items(), key=lambda x: -x[1])

    # Extract bigram phrases from descriptions using top words
    queries: List[str] = []
    seen_queries: Set[str] = set()
    top_word_set = {w for w, _ in top_words[:30]}

    for desc in descriptions:
        if not desc:
            continue
        phrases = _extract_key_phrases(desc, top_word_set, max_phrases_per_description)
        for phrase in phrases:
            phrase_lower = phrase.lower()
            if phrase_lower not in seen_queries and len(phrase.split()) >= 2:
                seen_queries.add(phrase_lower)
                queries.append(phrase)

    # If not enough bigram phrases, fall back to top individual words
    if len(queries) < max_queries:
        for word, _ in top_words:
            if word not in seen_queries and len(word) > 4:
                seen_queries.add(word)
                queries.append(word)
            if len(queries) >= max_queries:
                break

    return queries[:max_queries]


def _extract_content_words(text: str) -> List[str]:
    """Extract meaningful content words from text, filtering noise."""
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove common YouTube boilerplate patterns
    text = re.sub(r'(?i)subscribe\s+(to\s+)?my\s+channel', '', text)
    text = re.sub(r'(?i)follow\s+(me\s+)?on\s+\w+', '', text)
    # Split and clean
    words = text.lower().split()
    words = [re.sub(r'[^a-z0-9]', '', w) for w in words]
    return [w for w in words if len(w) > 3 and w not in _DESCRIPTION_STOP_WORDS]


def _extract_key_phrases(
    text: str,
    top_words: Set[str],
    max_phrases: int = 3,
) -> List[str]:
    """
    Extract short key phrases (bigrams/trigrams) from text using top-scored words.

    Args:
        text: Source text to extract phrases from.
        top_words: Set of high-scoring words to anchor phrases on.
        max_phrases: Maximum phrases to return.

    Returns:
        List of key phrase strings.
    """
    # Clean text
    text = re.sub(r'https?://\S+', '', text)
    words = text.split()
    cleaned = []
    for w in words:
        clean = re.sub(r'[^a-zA-Z0-9]', '', w).lower()
        if clean and len(clean) > 2 and clean not in _DESCRIPTION_STOP_WORDS:
            cleaned.append(clean)

    phrases: List[str] = []
    # Extract bigrams that contain at least one top word
    for i in range(len(cleaned) - 1):
        if cleaned[i] in top_words or cleaned[i + 1] in top_words:
            phrase = f"{cleaned[i]} {cleaned[i + 1]}"
            if phrase not in phrases:
                phrases.append(phrase)
        if len(phrases) >= max_phrases:
            break

    return phrases


# ============================================================================
# US-73-009: Video tag-derived search queries
# ============================================================================

# Generic tags that don't improve search specificity
_STOP_TAGS = {
    'video', 'youtube', 'official', 'channel', 'subscribe', 'like',
    'comment', 'share', 'watch', 'new', 'best', 'top', 'free',
    'live', 'full', 'hd', '4k', '1080p', '720p', 'shorts', 'vlog',
    'tutorial', 'how to', 'reaction', 'review', 'compilation',
    'highlights', 'episode', 'part', 'clip', 'trailer',
}


def extract_tags_from_nearby_matches(
    gap: 'GapSegment',
    locked: List['LockedMatch'],
    state: Any,
    max_tags: int = 3,
    max_distance: float = 120.0,
) -> List[str]:
    """
    Extract the most frequent video tags from locked matches near a gap.

    Finds locked matches within max_distance seconds of the gap, collects
    their video_tags from state.video_search_results, filters out generic
    stop-tags, and returns the top tags by frequency.

    Args:
        gap: The gap segment to find nearby tags for.
        locked: All locked matches in this pass.
        state: PipelineState with video_search_results.
        max_tags: Maximum tags to return (default 3).
        max_distance: Maximum timeline distance in seconds to consider
            a locked match as "nearby" (default 120s).

    Returns:
        List of tag strings, most frequent first, up to max_tags.
    """
    # Build video_id -> tags lookup from state
    vid_tags: Dict[str, List[str]] = {}
    for vsr in (state.video_search_results or []):
        vid_id = getattr(vsr, 'video_id', None) or (
            vsr.get('video_id') if isinstance(vsr, dict) else None
        )
        tags = getattr(vsr, 'video_tags', None) or (
            vsr.get('video_tags', []) if isinstance(vsr, dict) else []
        )
        if vid_id and tags:
            vid_tags[vid_id] = tags

    if not vid_tags:
        return []

    # Collect tags from nearby locked matches
    tag_freq: Dict[str, int] = defaultdict(int)
    for lock in locked:
        distance = abs(gap.position - lock.position)
        if distance > max_distance:
            continue
        tags = vid_tags.get(lock.video_id, [])
        for tag in tags:
            tag_lower = tag.lower().strip()
            if tag_lower and tag_lower not in _STOP_TAGS and len(tag_lower) >= 3:
                tag_freq[tag_lower] += 1

    if not tag_freq:
        return []

    # Sort by frequency descending, return top N
    sorted_tags = sorted(tag_freq.items(), key=lambda x: -x[1])
    return [tag for tag, _ in sorted_tags[:max_tags]]


# ============================================================================
# US-111-009: Context-aware iterative gap filling
# ============================================================================

# Stop words for context extraction
_CONTEXT_STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
    'these', 'those', 'it', 'its', 'they', 'them', 'their', 'we', 'us',
    'our', 'you', 'your', 'he', 'she', 'him', 'her', 'his', 'i', 'me', 'my',
}


@dataclass
class ContextSegment:
    """A matched segment used for context in gap-filling queries."""
    segment_index: int
    video_id: str
    title: str
    position: float  # Start time in timeline
    keywords: List[str] = field(default_factory=list)
    distance: float = 0.0  # Distance in seconds from the gap


@dataclass
class ContextQuery:
    """A query generated with context awareness."""
    query: str
    base_keywords: List[str] = field(default_factory=list)  # Keywords from gap text
    context_keywords: List[str] = field(default_factory=list)  # Keywords from nearby matches
    relevance_score: float = 0.0  # Weighted relevance score
    context_weight: float = 0.0  # Weight given to context (0-1)


def extract_context_from_nearby_matches(
    gap: GapSegment,
    locked_matches: List['LockedMatch'],
    state: Any,
    window_seconds: float = 180.0,
    max_context_segments: int = 5,
    topic_weight: float = 0.5,
) -> List[ContextSegment]:
    """
    Extract context from already-matched segments near a gap.

    US-111-009: This function finds locked/matched segments within the specified
    time window (default 3 minutes) of the gap and extracts their keywords/topics
    to inform gap-filling query generation.

    Args:
        gap: The gap segment to find context for.
        locked_matches: List of locked matches in the current pass.
        state: PipelineState with voiceover_segments and video metadata.
        window_seconds: Time window in seconds to look for context (default 180s).
        max_context_segments: Maximum number of context segments to return.
        topic_weight: Weight for topic relevance scoring (0-1).

    Returns:
        List of ContextSegment objects sorted by relevance (closest + most relevant first).
    """
    if not locked_matches:
        return []

    # Get voiceover text for keyword extraction from state
    voiceover_segments = getattr(state, 'voiceover_segments', []) or []

    # Find locked segments within the time window
    nearby_matches: List[ContextSegment] = []

    for lock in locked_matches:
        distance = abs(gap.position - lock.position)
        if distance > window_seconds:
            continue

        # Get keywords from the matched segment's voiceover
        keywords: List[str] = []
        if lock.segment_index < len(voiceover_segments):
            vo_segment = voiceover_segments[lock.segment_index]
            vo_text = getattr(vo_segment, 'text', '') or (
                vo_segment.get('text') if isinstance(vo_segment, dict) else ''
            )
            if vo_text:
                keywords = _extract_keywords_from_text(vo_text)

        nearby_matches.append(ContextSegment(
            segment_index=lock.segment_index,
            video_id=lock.video_id,
            title=lock.title or '',
            position=lock.position,
            keywords=keywords,
            distance=distance,
        ))

    # Sort by distance (closer = more relevant) and limit
    nearby_matches.sort(key=lambda x: (x.distance, -len(x.keywords)))
    return nearby_matches[:max_context_segments]


def _extract_keywords_from_text(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text for context matching.

    Args:
        text: Text to extract keywords from.
        max_keywords: Maximum number of keywords to return.

    Returns:
        List of keywords.
    """
    if not text:
        return []

    # Extract proper nouns (capitalized words)
    proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

    # Extract content words
    words = text.lower().split()
    words = [w.strip('.,!?;:"\'-()[]') for w in words]
    content_words = [
        w for w in words
        if len(w) > 3 and w not in _CONTEXT_STOP_WORDS
    ]

    # Combine and deduplicate
    all_keywords = list(set(proper_nouns + content_words))

    # Sort: proper nouns first, then by length (longer = more specific)
    proper_set = {kw.lower() for kw in proper_nouns}
    sorted_keywords = sorted(
        all_keywords,
        key=lambda kw: (kw.lower() not in proper_set, -len(kw))
    )

    return sorted_keywords[:max_keywords]


def compute_context_relevance(
    gap_keywords: List[str],
    context_keywords: List[str],
    topic_weight: float = 0.5,
) -> float:
    """
    Compute relevance score between gap keywords and context keywords.

    US-111-009: Uses weighted Jaccard similarity to score relevance.

    Args:
        gap_keywords: Keywords extracted from gap voiceover text.
        context_keywords: Keywords from nearby matched segments.
        topic_weight: Weight for topic overlap (0-1). Higher = more weight on topic similarity.

    Returns:
        Relevance score between 0 and 1.
    """
    if not gap_keywords or not context_keywords:
        return 0.0

    # Convert to lowercase sets for comparison
    gap_set = {kw.lower() for kw in gap_keywords}
    context_set = {kw.lower() for kw in context_keywords}

    # Jaccard similarity
    intersection = gap_set & context_set
    union = gap_set | context_set

    if not union:
        return 0.0

    jaccard = len(intersection) / len(union)

    # Also compute overlap ratio (what % of context keywords appear in gap)
    overlap_ratio = len(intersection) / len(context_set) if context_set else 0.0

    # Weighted combination
    score = (topic_weight * jaccard) + ((1 - topic_weight) * overlap_ratio)

    return min(score, 1.0)


def generate_context_aware_queries(
    gap: GapSegment,
    context_segments: List[ContextSegment],
    gap_keywords: List[str],
    max_queries: int = 3,
    context_boost: float = 0.15,
    topic_weight: float = 0.5,
) -> List[ContextQuery]:
    """
    Generate queries for a gap using context from nearby matched segments.

    US-111-009: This function combines gap text keywords with context keywords
    from nearby matched segments to create more informed search queries.

    Args:
        gap: The gap segment to generate queries for.
        context_segments: Context segments from nearby matches.
        gap_keywords: Keywords extracted from gap voiceover text.
        max_queries: Maximum number of queries to generate.
        context_boost: Boost weight for context-aware queries (0-1).
        topic_weight: Weight for topic relevance scoring.

    Returns:
        List of ContextQuery objects sorted by relevance score.
    """
    if not gap_keywords:
        return []

    queries: List[ContextQuery] = []

    # 1. Base query: just gap keywords
    base_query_text = ' '.join(gap_keywords[:5])
    base_context_keywords = []

    # Calculate relevance between gap and each context segment
    total_relevance = 0.0
    total_context_keywords: List[str] = []

    for ctx_seg in context_segments:
        relevance = compute_context_relevance(
            gap_keywords,
            ctx_seg.keywords,
            topic_weight
        )
        total_relevance += relevance
        total_context_keywords.extend(ctx_seg.keywords)

    avg_relevance = total_relevance / len(context_segments) if context_segments else 0.0
    unique_context_keywords = list(dict.fromkeys(total_context_keywords))  # Preserve order, dedupe

    # 2. Add base query (no context)
    queries.append(ContextQuery(
        query=base_query_text,
        base_keywords=gap_keywords[:5],
        context_keywords=[],
        relevance_score=1.0,  # Base query always has full relevance
        context_weight=0.0,
    ))

    # 3. If we have relevant context, add context-enhanced queries
    if avg_relevance > 0.1 and unique_context_keywords:
        # Find context keywords not already in gap keywords
        gap_set = {kw.lower() for kw in gap_keywords}
        new_context_keywords = [
            kw for kw in unique_context_keywords
            if kw.lower() not in gap_set
        ]

        # Add context query with novel keywords
        if new_context_keywords:
            # Combine top gap keywords with novel context keywords
            combined_keywords = gap_keywords[:3] + new_context_keywords[:3]
            combined_query = ' '.join(combined_keywords)

            # Score: base relevance minus boost penalty (context might reduce specificity)
            context_score = avg_relevance * (1.0 - context_boost * 0.5)

            queries.append(ContextQuery(
                query=combined_query,
                base_keywords=gap_keywords[:3],
                context_keywords=new_context_keywords[:3],
                relevance_score=context_score,
                context_weight=context_boost,
            ))

        # Add context-only query (from nearby matches) for fallback
        if len(unique_context_keywords) >= 2:
            context_only_query = ' '.join(unique_context_keywords[:4])
            context_score = avg_relevance * (1.0 - context_boost)

            queries.append(ContextQuery(
                query=context_only_query,
                base_keywords=[],
                context_keywords=unique_context_keywords[:4],
                relevance_score=context_score,
                context_weight=context_boost,
            ))

    # Sort by relevance score descending
    queries.sort(key=lambda q: -q.relevance_score)

    return queries[:max_queries]


def get_context_keywords_for_gap(
    gap: GapSegment,
    locked_matches: List['LockedMatch'],
    state: Any,
    window_seconds: float = 180.0,
    max_keywords: int = 5,
) -> List[str]:
    """
    Convenience function to get context keywords for a gap.

    This is a simpler interface that returns just the keyword list,
    useful when you don't need the full ContextQuery objects.

    Args:
        gap: The gap segment to get context for.
        locked_matches: List of locked matches.
        state: PipelineState.
        window_seconds: Time window for context.
        max_keywords: Maximum keywords to return.

    Returns:
        List of context keywords.
    """
    context_segments = extract_context_from_nearby_matches(
        gap=gap,
        locked_matches=locked_matches,
        state=state,
        window_seconds=window_seconds,
        max_context_segments=5,
    )

    # Collect all keywords from context segments
    all_keywords: List[str] = []
    for ctx_seg in context_segments:
        all_keywords.extend(ctx_seg.keywords)

    # Deduplicate while preserving order
    seen = set()
    unique_keywords = []
    for kw in all_keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append(kw)

    return unique_keywords[:max_keywords]
