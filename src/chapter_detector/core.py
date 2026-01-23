"""
Core chapter detection logic using LLM.
"""

import json
import logging
import re
from typing import List, Optional, Callable

from .models import Chapter, ChapterDetectionResult
from .prompts import LISTICLE_DETECTION_PROMPT

logger = logging.getLogger(__name__)


class ChapterDetector:
    """
    Detects chapters/list items in voiceover using LLM.

    Handles:
    - Listicle/ranking videos (Number 15, Number 14, ...)
    - ASR transcription error correction
    - Per-chapter keyword extraction
    """

    def __init__(self, llm_call_fn: Callable[[str], str]):
        """
        Initialize with LLM call function.

        Args:
            llm_call_fn: Function that takes prompt string and returns LLM response string
        """
        self.llm_call_fn = llm_call_fn

    def detect(self, segments: List[dict], topic: str = "") -> ChapterDetectionResult:
        """
        Detect chapters in voiceover segments.

        Args:
            segments: List of segment dicts with 'index', 'text', 'start', 'end' keys
            topic: Documentary topic for keyword context (e.g., "dog emotions")

        Returns:
            ChapterDetectionResult with detected chapters
        """
        if not segments:
            return ChapterDetectionResult(
                is_listicle=False,
                list_type=None,
                total_items=0,
                chapters=[]
            )

        # Build indexed text for LLM
        indexed_text = self._build_indexed_text(segments)

        # Call LLM for detection
        prompt = LISTICLE_DETECTION_PROMPT.format(
            indexed_text=indexed_text,
            topic=topic or "general documentary"
        )

        logger.info("Calling LLM for chapter detection...")
        try:
            response = self.llm_call_fn(prompt)
            result = self._parse_response(response, segments)
            logger.info(f"Chapter detection result: {result}")
            return result
        except Exception as e:
            logger.error(f"Chapter detection failed: {e}")
            return ChapterDetectionResult(
                is_listicle=False,
                list_type=None,
                total_items=0,
                chapters=[]
            )

    def _build_indexed_text(self, segments: List[dict]) -> str:
        """Build text with segment indices for LLM."""
        lines = []
        for seg in segments:
            idx = seg.get('index', 0)
            text = seg.get('text', '')
            lines.append(f"[{idx}] {text}")
        return "\n".join(lines)

    def _parse_response(self, response: str, segments: List[dict]) -> ChapterDetectionResult:
        """Parse LLM response into ChapterDetectionResult."""
        # Extract JSON from response
        response = response.strip()

        # Find JSON object
        start = response.find('{')
        end = response.rfind('}') + 1

        if start < 0 or end <= start:
            logger.warning("No JSON found in response")
            return ChapterDetectionResult(
                is_listicle=False,
                list_type=None,
                total_items=0,
                chapters=[]
            )

        json_str = response[start:end]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Raw response: {json_str[:500]}")
            return ChapterDetectionResult(
                is_listicle=False,
                list_type=None,
                total_items=0,
                chapters=[]
            )

        is_listicle = data.get('is_listicle', False)
        list_type = data.get('list_type')
        total_items = data.get('total_items', 0)
        intro_end = data.get('intro_end_segment')
        items = data.get('items', [])

        if not is_listicle or not items:
            return ChapterDetectionResult(
                is_listicle=is_listicle,
                list_type=list_type,
                total_items=total_items,
                chapters=[]
            )

        # Convert items to Chapter objects
        chapters = []
        max_segment = len(segments) - 1

        for i, item in enumerate(items):
            name = item.get('name', '')
            corrected_name = item.get('corrected_name', name)
            rank = item.get('rank')
            start_segment = item.get('start_segment', 0)
            keywords = item.get('keywords', [])

            # Calculate end_segment (start of next item - 1, or max)
            if i + 1 < len(items):
                next_start = items[i + 1].get('start_segment', max_segment)
                end_segment = max(start_segment, next_start - 1)
            else:
                end_segment = max_segment

            # Ensure valid range
            start_segment = max(0, min(start_segment, max_segment))
            end_segment = max(start_segment, min(end_segment, max_segment))

            # Generate default keywords if none provided
            if not keywords:
                keywords = self._generate_default_keywords(corrected_name)

            chapter = Chapter(
                name=name,
                corrected_name=corrected_name,
                rank=rank,
                start_segment=start_segment,
                end_segment=end_segment,
                keywords=keywords
            )
            chapters.append(chapter)
            logger.info(f"  Chapter: {chapter}")

        return ChapterDetectionResult(
            is_listicle=is_listicle,
            list_type=list_type,
            total_items=total_items,
            chapters=chapters,
            intro_end_segment=intro_end
        )

    def _generate_default_keywords(self, entity_name: str) -> List[str]:
        """Generate default search keywords for an entity."""
        return [
            f"{entity_name} restaurant",
            f"{entity_name} exterior",
            f"{entity_name} sign",
            f"{entity_name} food",
            f"{entity_name} storefront"
        ]


def detect_chapters(
    segments: List[dict],
    llm_call_fn: Callable[[str], str],
    topic: str = ""
) -> ChapterDetectionResult:
    """
    Convenience function for chapter detection.

    Args:
        segments: List of segment dicts
        llm_call_fn: LLM call function
        topic: Documentary topic for keyword context

    Returns:
        ChapterDetectionResult
    """
    detector = ChapterDetector(llm_call_fn)
    return detector.detect(segments, topic=topic)
