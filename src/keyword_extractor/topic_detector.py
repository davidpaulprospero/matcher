"""
Topic detection for video content.

Identifies primary topic/category (e.g., "travel", "science", "history").
"""

import logging
from typing import Optional

from .prompts import TOPIC_DETECTION_PROMPT

logger = logging.getLogger(__name__)


def detect_topic(
    text: str,
    llm_client,
    max_text_length: int = 3000
) -> str:
    """
    Detect the main topic from the text using LLM.

    Args:
        text: Text to analyze for topic detection
        llm_client: LLM client instance (must have generate() method)
        max_text_length: Maximum characters to send to LLM

    Returns:
        Topic string (2-5 words), or empty string if detection fails
    """
    logger.debug("=== TOPIC DETECTION STARTING ===")

    # Use LLM-based topic detection (required for accurate topic detection)
    if llm_client:
        topic = detect_topic_llm(text[:max_text_length], llm_client)
        if topic:
            return topic
        else:
            logger.warning("LLM topic detection returned empty result")
    else:
        logger.warning("No LLM client configured - topic detection disabled")

    # Return empty if LLM not available or failed
    return ""


def detect_topic_llm(text: str, llm_client) -> str:
    """
    Use LLM to detect the main topic.

    Args:
        text: Text to analyze (should be pre-truncated to ~2000-3000 chars)
        llm_client: LLM client instance

    Returns:
        Topic string (2-5 words), or empty string if detection fails
    """
    prompt = TOPIC_DETECTION_PROMPT.format(text=text[:2000])

    try:
        from src.llm_client import LLMRequest, ResponseFormat

        request = LLMRequest(
            prompt=prompt,
            response_format=ResponseFormat.TEXT,
            max_tokens=100,
            cache_key_prefix="topic_detection"
        )
        response = llm_client.generate(request)

        topic = response.text.strip().strip('"').strip("'")
        logger.debug(f"LLM raw response text: '{topic[:100]}'")

        if topic and len(topic) < 100:  # Sanity check
            logger.info(f"LLM detected topic: {topic}")
            return topic
        else:
            logger.warning(f"LLM topic too long or empty: '{topic[:50]}...'")
    except Exception as e:
        logger.warning(f"LLM topic detection failed: {e}", exc_info=True)

    return ""
