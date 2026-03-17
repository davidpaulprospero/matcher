"""
Per-segment keyword extraction.

Supports two extraction modes:
1. Per-segment: 1 keyword per segment
2. Grouped: 1 search query per N segments (default 3)
"""

import json
import re
import logging
from typing import List, Dict

from .prompts import BATCH_SEGMENT_KEYWORDS_PROMPT, GROUPED_SEGMENT_KEYWORDS_PROMPT

logger = logging.getLogger(__name__)


def extract_keyword_per_segment(
    segments: List[Dict],
    llm_client,
    llm_call_function,
    topic: str = ""
) -> List[str]:
    """
    Extract ONE keyword per segment for precise B-roll matching.

    Args:
        segments: List of segment dicts with 'text' key
        llm_client: LLM client instance (for availability check)
        llm_call_function: Function to call LLM (signature: prompt -> str)
        topic: Documentary topic for context

    Returns:
        List of keywords (one per segment, in order)
    """
    if not llm_client:
        logger.warning("No LLM client - using simple keyword extraction")
        return simple_segment_keywords(segments, topic)

    # Prepare segments text
    segment_texts = []
    for i, seg in enumerate(segments):
        text = seg.get('text', '') if isinstance(seg, dict) else getattr(seg, 'text', '')
        text = text.strip()
        if text:
            segment_texts.append(f"{i+1}. \"{text}\"")
        else:
            segment_texts.append(f"{i+1}. [empty]")

    # Process in batches to avoid token limits
    batch_size = 50
    all_keywords = []

    for batch_start in range(0, len(segment_texts), batch_size):
        batch_end = min(batch_start + batch_size, len(segment_texts))
        batch = segment_texts[batch_start:batch_end]

        prompt = BATCH_SEGMENT_KEYWORDS_PROMPT.format(
            topic=topic or "documentary",
            segments_text="\n".join(batch)
        )

        try:
            response = llm_call_function(prompt)
            response = response.strip()

            # Parse JSON array
            start = response.find('[')
            end = response.rfind(']') + 1

            if start >= 0 and end > start:
                json_str = response[start:end]
                keywords = json.loads(json_str)

                # Ensure we have one keyword per segment in batch
                while len(keywords) < (batch_end - batch_start):
                    keywords.append(topic or "documentary footage")

                all_keywords.extend(keywords[:batch_end - batch_start])
            else:
                # Fallback for this batch
                logger.warning(f"Could not parse batch {batch_start}-{batch_end}, using fallback")
                for i in range(batch_start, batch_end):
                    seg_text = segments[i].get('text', '') if isinstance(segments[i], dict) else getattr(segments[i], 'text', '')
                    all_keywords.append(extract_simple_keyword(seg_text, topic))

        except Exception as e:
            logger.warning(f"Batch keyword extraction failed: {e}")
            # Fallback for this batch
            for i in range(batch_start, batch_end):
                seg_text = segments[i].get('text', '') if isinstance(segments[i], dict) else getattr(segments[i], 'text', '')
                all_keywords.append(extract_simple_keyword(seg_text, topic))

    return all_keywords


def simple_segment_keywords(segments: List[Dict], topic: str) -> List[str]:
    """
    Simple fallback: extract keywords without LLM.

    Args:
        segments: List of segment dicts with 'text' key
        topic: Documentary topic for context

    Returns:
        List of keywords (one per segment)
    """
    keywords = []
    for seg in segments:
        text = seg.get('text', '') if isinstance(seg, dict) else getattr(seg, 'text', '')
        keywords.append(extract_simple_keyword(text, topic))
    return keywords


def extract_simple_keyword(text: str, topic: str) -> str:
    """
    Extract a simple keyword from segment text.

    Args:
        text: Segment text
        topic: Documentary topic (used as fallback)

    Returns:
        Keyword string (3-4 meaningful words)
    """
    if not text or len(text.strip()) < 10:
        return topic or "documentary footage"

    # Remove common stopwords and get key terms
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
        'because', 'until', 'while', 'although', 'though', 'this', 'that',
        'these', 'those', 'what', 'which', 'who', 'whom', 'whose', 'it', 'its',
        'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your', 'he', 'him',
        'his', 'she', 'her', 'i', 'me', 'my'
    }

    # Clean and tokenize
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    # Filter stopwords and get unique meaningful words
    meaningful = []
    seen = set()
    for word in words:
        if word not in stopwords and word not in seen:
            meaningful.append(word)
            seen.add(word)

    # Take top 3-4 words
    if meaningful:
        keyword = ' '.join(meaningful[:4])
        # Capitalize proper nouns (words that were capitalized in original)
        original_words = text.split()
        for orig_word in original_words:
            if orig_word[0].isupper() and orig_word.lower() in keyword.lower():
                keyword = keyword.replace(orig_word.lower(), orig_word)
        return keyword

    return topic or "documentary footage"


def extract_keywords_grouped(
    segments: List[Dict],
    llm_client,
    llm_call_function,
    topic: str = "",
    segments_per_query: int = 3
) -> List[str]:
    """
    Extract ONE search query per group of N segments.

    Groups segments into batches and generates one unified search query
    per batch, reducing total queries while maintaining context.

    Args:
        segments: List of segment dicts with 'text' key
        llm_client: LLM client instance (for availability check)
        llm_call_function: Function to call LLM (signature: prompt -> str)
        topic: Documentary topic for context
        segments_per_query: Number of segments per search query (default: 3)

    Returns:
        List of search queries (one per segment group)
    """
    if not segments:
        return []

    if not llm_client:
        logger.warning("No LLM client - using simple grouped extraction")
        return simple_grouped_keywords(segments, topic, segments_per_query)

    keywords = []
    total_groups = (len(segments) + segments_per_query - 1) // segments_per_query

    for group_idx in range(0, len(segments), segments_per_query):
        group_end = min(group_idx + segments_per_query, len(segments))
        group = segments[group_idx:group_end]
        group_num = (group_idx // segments_per_query) + 1

        # Combine segment texts for this group
        segment_texts = []
        for i, seg in enumerate(group):
            text = seg.get('text', '') if isinstance(seg, dict) else getattr(seg, 'text', '')
            text = text.strip()
            if text:
                segment_texts.append(f"{i+1}. \"{text}\"")
            else:
                segment_texts.append(f"{i+1}. [empty]")

        prompt = GROUPED_SEGMENT_KEYWORDS_PROMPT.format(
            topic=topic or "documentary",
            segment_count=len(group),
            segments_text="\n".join(segment_texts)
        )

        try:
            response = llm_call_function(prompt)
            response = response.strip()

            # Clean up response - remove quotes, extra whitespace
            keyword = response.strip('"\'').strip()

            # Validate: should be 2-8 words, not empty
            word_count = len(keyword.split())
            if keyword and 2 <= word_count <= 8:
                keywords.append(keyword)
                logger.debug(f"Group {group_num}/{total_groups}: '{keyword}'")
            else:
                # Fallback for this group
                fallback = simple_grouped_keyword(group, topic)
                keywords.append(fallback)
                logger.warning(f"Group {group_num}: Invalid response, using fallback '{fallback}'")

        except Exception as e:
            logger.warning(f"Group {group_num} extraction failed: {e}")
            fallback = simple_grouped_keyword(group, topic)
            keywords.append(fallback)

    logger.info(f"Grouped extraction: {len(segments)} segments → {len(keywords)} search queries "
                f"({segments_per_query} segments per query)")

    return keywords


def simple_grouped_keywords(
    segments: List[Dict],
    topic: str,
    segments_per_query: int = 3
) -> List[str]:
    """
    Simple fallback: extract grouped keywords without LLM.

    Args:
        segments: List of segment dicts with 'text' key
        topic: Documentary topic for context
        segments_per_query: Number of segments per search query

    Returns:
        List of keywords (one per segment group)
    """
    keywords = []

    for group_idx in range(0, len(segments), segments_per_query):
        group_end = min(group_idx + segments_per_query, len(segments))
        group = segments[group_idx:group_end]
        keywords.append(simple_grouped_keyword(group, topic))

    return keywords


def simple_grouped_keyword(segments: List[Dict], topic: str) -> str:
    """
    Extract a simple keyword from a group of segments.

    Args:
        segments: List of segment dicts with 'text' key
        topic: Documentary topic (used as fallback)

    Returns:
        Keyword string covering the segment group
    """
    # Combine all text from the group
    combined_text = []
    for seg in segments:
        text = seg.get('text', '') if isinstance(seg, dict) else getattr(seg, 'text', '')
        if text:
            combined_text.append(text.strip())

    if not combined_text:
        return topic or "documentary footage"

    full_text = " ".join(combined_text)
    return extract_simple_keyword(full_text, topic)
