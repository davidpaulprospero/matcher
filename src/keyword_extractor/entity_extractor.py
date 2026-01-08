"""
Named entity extraction and conversion to visual keywords.

Extracts PERSON, PLACE, ORG, DATE, EVENT entities from LLM response
and converts them to filmable search keywords.
"""

import json
import logging
from typing import List, Tuple, Dict

from .prompts import ENTITY_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


def extract_entities(
    text: str,
    topic: str,
    llm_call_function,
) -> Tuple[List[str], List[Dict]]:
    """
    Extract named entities and convert to visual keywords.

    Args:
        text: Text to extract entities from
        topic: Topic context for extraction
        llm_call_function: Function to call LLM (signature: prompt -> str)

    Returns:
        Tuple of (keywords, raw_entities)
        - keywords: List of visual search keywords derived from entities
        - raw_entities: List of dicts with entity metadata
    """
    prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
    raw_entities = []

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

            # Deduplicate keywords while preserving order
            seen = set()
            unique_keywords = []
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower not in seen:
                    seen.add(kw_lower)
                    unique_keywords.append(kw)

            return unique_keywords, raw_entities

    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")

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
