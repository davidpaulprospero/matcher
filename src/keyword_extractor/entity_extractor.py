"""
Named entity extraction and conversion to visual keywords.

Extracts PERSON, PLACE, ORG, DATE, EVENT entities from LLM response
and converts them to filmable search keywords.

Also includes pattern-based detection for listicle content:
- "Number 15, Denny's" -> Entity: Denny's
- "Coming in at #10, Little Caesars" -> Entity: Little Caesars
"""

import json
import logging
import re
from typing import List, Tuple, Dict, Optional

from .prompts import ENTITY_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


def detect_listicle_entities(text: str) -> List[Dict]:
    """
    Detect entities from listicle/ranking patterns in text.

    Patterns detected:
    - "Number 15, Denny's" or "number fifteen, Denny's"
    - "Coming in at number 10, Little Caesars"
    - "#8 on our list is Jack in the Box"
    - "At number 7, we have Popeyes"
    - "In 5th place, Burger King"

    Args:
        text: Full voiceover text

    Returns:
        List of entity dicts with 'text', 'type', 'search_keyword', 'context'
    """
    entities = []
    seen_names = set()

    # Number word to digit mapping for word-based patterns
    number_words = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
        'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10
    }
    number_words_pattern = '|'.join(number_words.keys())

    # Entity name pattern - captures 1-4 capitalized words with apostrophes/hyphens
    # Greedy capture until we hit a period followed by space+capital (next sentence)
    entity_pattern = r"([A-Z][A-Za-z']+(?:\s+[A-Za-z']+){0,3})"

    # Pattern 1: "Number X, EntityName." or "Number X, EntityName. Next sentence"
    # Captures up to the period before next sentence
    pattern1 = rf'[Nn]umber\s+(\d+|{number_words_pattern})[,.\s]+{entity_pattern}(?=\.|\s+[A-Z]|\s*$)'

    # Pattern 2: "#X EntityName"
    pattern2 = rf'#\s*(\d+)[,.\s]+{entity_pattern}(?=\.|\s+[A-Z]|\s*$)'

    # Pattern 3: "Coming in at number X, EntityName"
    pattern3 = rf'[Cc]oming\s+in\s+at\s+(?:number\s+)?(\d+|{number_words_pattern})[,.\s]+{entity_pattern}(?=\.|\s+[A-Z]|\s*$)'

    # Pattern 4: "At number X, we have EntityName"
    pattern4 = rf'[Aa]t\s+(?:number\s+)?(\d+|{number_words_pattern})[,.\s]+(?:we\s+have\s+)?{entity_pattern}(?=\.|\s+[A-Z]|\s*$)'

    # Pattern 5: "In Xth place, EntityName"
    pattern5 = rf'[Ii]n\s+(\d+)(?:st|nd|rd|th)\s+place[,.\s]+{entity_pattern}(?=\.|\s+[A-Z]|\s*$)'

    # Pattern 6: "X on our list is EntityName"
    pattern6 = rf'(\d+)\s+on\s+our\s+list\s+is\s+{entity_pattern}(?=\.|\s+[A-Z]|\s*$)'

    # Pattern 7: "And number X, EntityName" (for "And number 1, McDonald's")
    pattern7 = rf'[Aa]nd\s+[Nn]umber\s+(\d+|{number_words_pattern})[,.\s]+{entity_pattern}(?=\.|\s+[A-Z]|\s*$)'

    all_patterns = [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7]

    for pattern in all_patterns:
        for match in re.finditer(pattern, text):
            entity_name = match.group(2).strip()

            # Clean up entity name - remove trailing punctuation and common words
            entity_name = re.sub(r'[,.\s]+$', '', entity_name)
            # Remove common trailing words that aren't part of the brand name
            entity_name = re.sub(r'\s+(is|was|has|have|will|can|the|Once|The|Despite|Fast|Premium|Even|Kentucky|Fried|Chicken)$', '', entity_name, flags=re.IGNORECASE)
            entity_name = entity_name.strip()

            # Skip if too short or already seen
            if len(entity_name) < 3:
                continue
            if entity_name.lower() in seen_names:
                continue

            seen_names.add(entity_name.lower())

            # Generate search keyword
            search_keyword = f"{entity_name} restaurant review"  # Default for restaurants
            if any(word in text.lower() for word in ['restaurant', 'food', 'chain', 'dining', 'menu', 'pizza', 'chicken', 'burger', 'fast food']):
                search_keyword = f"{entity_name} restaurant food tour"
            elif any(word in text.lower() for word in ['store', 'retail', 'shop']):
                search_keyword = f"{entity_name} store walkthrough"

            entities.append({
                'text': entity_name,
                'type': 'ORG',
                'search_keyword': search_keyword,
                'context': 'listicle item'
            })
            logger.info(f"Listicle pattern detected: {entity_name}")

    return entities


def create_manual_entities(manual_names: List[str], topic: str = "") -> List[Dict]:
    """
    Create entity dicts from manual entity names.

    Args:
        manual_names: List of entity names to create
        topic: Topic context for keyword generation

    Returns:
        List of entity dicts
    """
    entities = []
    topic_lower = topic.lower() if topic else ""

    for name in manual_names:
        name = name.strip()
        if not name:
            continue

        # Generate contextual search keyword
        if any(word in topic_lower for word in ['restaurant', 'food', 'chain', 'dining', 'fast food']):
            search_keyword = f"{name} restaurant review"
        elif any(word in topic_lower for word in ['store', 'retail', 'shop']):
            search_keyword = f"{name} store tour"
        else:
            search_keyword = f"{name} footage"

        entities.append({
            'text': name,
            'type': 'ORG',
            'search_keyword': search_keyword,
            'context': 'manual override'
        })
        logger.info(f"Manual entity added: {name}")

    return entities


def extract_entities(
    text: str,
    topic: str,
    llm_call_function,
    manual_entities: Optional[List[str]] = None,
    detect_listicle: bool = True,
) -> Tuple[List[str], List[Dict]]:
    """
    Extract named entities and convert to visual keywords.

    Combines three sources:
    1. LLM extraction (primary)
    2. Listicle pattern detection (backup for "Number X, Entity" patterns)
    3. Manual entity override (config-provided list)

    Args:
        text: Text to extract entities from
        topic: Topic context for extraction
        llm_call_function: Function to call LLM (signature: prompt -> str)
        manual_entities: Optional list of entity names to always include
        detect_listicle: Whether to detect "Number X, Entity" patterns

    Returns:
        Tuple of (keywords, raw_entities)
        - keywords: List of visual search keywords derived from entities
        - raw_entities: List of dicts with entity metadata
    """
    prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
    raw_entities = []
    seen_entity_names = set()

    # Phase 1: Detect listicle patterns (run first to inform LLM results)
    listicle_entities = []
    if detect_listicle:
        listicle_entities = detect_listicle_entities(text)
        for ent in listicle_entities:
            seen_entity_names.add(ent['text'].lower())
        if listicle_entities:
            logger.info(f"Detected {len(listicle_entities)} listicle entities from patterns")

    # Phase 2: Add manual entities (config override)
    manual_entity_list = []
    if manual_entities:
        manual_entity_list = create_manual_entities(manual_entities, topic)
        for ent in manual_entity_list:
            seen_entity_names.add(ent['text'].lower())
        if manual_entity_list:
            logger.info(f"Added {len(manual_entity_list)} manual entities from config")

    try:
        response = llm_call_function(prompt)

        # Try to parse JSON response
        response = response.strip()

        # Find JSON object in response
        start = response.find('{')
        end = response.rfind('}') + 1

        if start >= 0 and end > start:
            json_str = response[start:end]
            entities = json.loads(json_str)

            keywords = []

            # Extract search keywords from each entity type
            # Process people entities
            for person in entities.get('people', []):
                kw = person.get('search_keyword', '')
                name = person.get('name', '')
                if kw and len(kw) > 3:
                    keywords.append(kw)
                if name:
                    raw_entities.append({
                        'text': name,
                        'type': 'PERSON',
                        'search_keyword': kw,
                        'context': person.get('context', '')
                    })

            # Process place entities
            for place in entities.get('places', []):
                kw = place.get('search_keyword', '')
                name = place.get('name', '')
                if kw and len(kw) > 3:
                    keywords.append(kw)
                if name:
                    raw_entities.append({
                        'text': name,
                        'type': 'GPE',  # Geo-political entity
                        'search_keyword': kw,
                        'context': place.get('context', '')
                    })

            # Process organization entities
            for org in entities.get('organizations', []):
                kw = org.get('search_keyword', '')
                name = org.get('name', '')
                if kw and len(kw) > 3:
                    keywords.append(kw)
                if name:
                    raw_entities.append({
                        'text': name,
                        'type': 'ORG',
                        'search_keyword': kw,
                        'context': org.get('context', '')
                    })

            # Process date entities
            for date in entities.get('dates', []):
                kw = date.get('search_keyword', '')
                name = date.get('date', date.get('name', ''))
                if kw and len(kw) > 3:
                    keywords.append(kw)
                if name:
                    raw_entities.append({
                        'text': name,
                        'type': 'DATE',
                        'search_keyword': kw,
                        'context': date.get('context', '')
                    })

            # Process event entities
            for event in entities.get('events', []):
                kw = event.get('search_keyword', '')
                name = event.get('name', '')
                if kw and len(kw) > 3:
                    keywords.append(kw)
                if name:
                    raw_entities.append({
                        'text': name,
                        'type': 'EVENT',
                        'search_keyword': kw,
                        'context': event.get('context', '')
                    })

            # Phase 3: Merge with listicle and manual entities (add those not found by LLM)
            llm_entity_names = {ent['text'].lower() for ent in raw_entities}

            # Add listicle entities not found by LLM
            for ent in listicle_entities:
                if ent['text'].lower() not in llm_entity_names:
                    raw_entities.append(ent)
                    if ent.get('search_keyword'):
                        keywords.append(ent['search_keyword'])
                    logger.debug(f"Added listicle entity not found by LLM: {ent['text']}")

            # Add manual entities not found by LLM or listicle
            all_names = {ent['text'].lower() for ent in raw_entities}
            for ent in manual_entity_list:
                if ent['text'].lower() not in all_names:
                    raw_entities.append(ent)
                    if ent.get('search_keyword'):
                        keywords.append(ent['search_keyword'])
                    logger.debug(f"Added manual entity: {ent['text']}")

            # Deduplicate keywords while preserving order
            seen = set()
            unique_keywords = []
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower not in seen:
                    seen.add(kw_lower)
                    unique_keywords.append(kw)

            logger.info(f"Entity extraction complete: {len(raw_entities)} entities, {len(unique_keywords)} keywords")
            return unique_keywords, raw_entities

    except Exception as e:
        logger.warning(f"LLM entity extraction failed: {e}")

    # Fallback: use listicle and manual entities even if LLM failed
    fallback_entities = listicle_entities + manual_entity_list
    fallback_keywords = [ent['search_keyword'] for ent in fallback_entities if ent.get('search_keyword')]

    if fallback_entities:
        logger.info(f"Using fallback entities: {len(fallback_entities)} from patterns/manual")
        return fallback_keywords, fallback_entities

    return [], []


def parse_entity_json(response: str) -> Dict:
    """
    Parse JSON entity response from LLM.

    Args:
        response: Raw LLM response containing JSON

    Returns:
        Parsed entities dict with keys: people, places, organizations, dates, events
    """
    response = response.strip()

    # Find JSON object in response
    start = response.find('{')
    end = response.rfind('}') + 1

    if start >= 0 and end > start:
        json_str = response[start:end]
        return json.loads(json_str)

    return {}
