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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from ..state import PipelineState, VoiceoverSegment

logger = logging.getLogger(__name__)


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
    pattern_type: str = ""  # Detected pattern
    keywords: List[str] = field(default_factory=list)


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


def extract_keywords_for_gap(gap: GapSegment, max_keywords: int = 5) -> List[str]:
    """
    Extract search-relevant keywords from a gap segment.

    Prioritizes:
    1. Proper nouns (capitalized words)
    2. Pattern-specific words (actions, locations)
    3. Content words (nouns, verbs, adjectives)

    Args:
        gap: GapSegment to extract keywords from
        max_keywords: Maximum keywords to return

    Returns:
        List of keywords suitable for search queries
    """
    text = gap.voiceover_text
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

    # Deduplicate while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            unique_keywords.append(kw)

    return unique_keywords[:max_keywords]
