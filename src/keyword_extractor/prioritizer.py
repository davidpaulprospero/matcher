"""
Keyword prioritization with scoring.

Assigns priority scores (0.0-1.0) to keywords based on:
- Entity vs general keyword
- Mention frequency
- Topic relevance
- Visual specificity
"""

import logging
from typing import List, Dict

from .models import PrioritizedKeyword

logger = logging.getLogger(__name__)


def build_prioritized_keywords(
    keywords: List[str],
    entity_keywords: List[str],
    raw_entities: List[Dict],
    full_text: str,
    topic: str
) -> List[PrioritizedKeyword]:
    """
    Build prioritized keyword list based on importance signals.

    Priority factors:
    - Entity-based keywords (people, places) get higher priority (0.9 base)
    - Keywords matching topic get boost (0.8 base)
    - Frequency of mention in script (up to +0.1)
    - Visual specificity (4K, drone, footage suffixes) (+0.05)

    Args:
        keywords: Final validated keywords
        entity_keywords: Keywords derived from named entities
        raw_entities: Raw entity data from extraction
        full_text: Original voiceover text
        topic: Detected topic

    Returns:
        List of PrioritizedKeyword sorted by priority (highest first)
    """
    prioritized = []
    text_lower = full_text.lower()
    topic_lower = topic.lower() if topic else ""

    # Build entity name set for quick lookup
    entity_names = set()
    for entity in raw_entities:
        name = entity.get('name', '').lower()
        if name:
            entity_names.add(name)

    for kw in keywords:
        kw_lower = kw.lower()

        # Determine source and base priority
        if kw in entity_keywords or any(name in kw_lower for name in entity_names):
            source = 'entity'
            base_priority = 0.9
        elif topic_lower and any(word in kw_lower for word in topic_lower.split()):
            source = 'topic'
            base_priority = 0.8
        else:
            source = 'general'
            base_priority = 0.5

        # Count mentions (approximate)
        mention_count = text_lower.count(kw_lower.split()[0]) if kw_lower.split() else 1
        mention_count = min(mention_count, 20)  # Cap at 20

        # Mention boost: more mentions = higher priority
        mention_boost = min(0.1, mention_count * 0.01)  # Up to +0.1

        # Visual specificity boost
        specificity_boost = 0.0
        if any(term in kw_lower for term in ['4k', 'drone', 'aerial', 'footage', 'timelapse']):
            specificity_boost = 0.05

        # Calculate final priority
        priority = min(1.0, base_priority + mention_boost + specificity_boost)

        prioritized.append(PrioritizedKeyword(
            keyword=kw,
            priority=priority,
            source=source,
            mention_count=mention_count
        ))

    # Sort by priority (descending)
    prioritized.sort(key=lambda x: x.priority, reverse=True)

    # Log priority breakdown
    if prioritized:
        entity_count = sum(1 for pk in prioritized if pk.source == 'entity')
        topic_count = sum(1 for pk in prioritized if pk.source == 'topic')
        general_count = sum(1 for pk in prioritized if pk.source == 'general')
        logger.info(f"Keyword priorities: {entity_count} entity, {topic_count} topic, {general_count} general")
        logger.info(f"[KEYWORD_EXTRACT] Prioritization complete: {len(prioritized)} keywords ranked")

    return prioritized
