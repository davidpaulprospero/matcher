"""
LLM provider implementations for matching.

Migrated from matching.py lines 228-484.
All providers use unified src/llm_client/ (Rule 9).

Providers:
- LLMProvider: Abstract base class
- GeminiMatcher: Google Gemini Flash (primary provider)
- ClaudeMatcher: Anthropic Claude Haiku (secondary/ambiguous)
- LocalLLMMatcher: Ollama (local LLM, one-at-a-time processing)

Utilities:
- validate_llm_reasoning: Check if LLM reasoning is specific (not generic)
- select_negative_sample: Select a negative sample from bottom candidates
"""

import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Set

from ..utils import SRTSegment

logger = logging.getLogger(__name__)


# Generic reasoning phrases that indicate low-quality LLM output
GENERIC_REASONING_PHRASES = frozenset([
    "good match",
    "best match",
    "matches well",
    "relevant",
    "related",
    "similar",
    "appropriate",
    "suitable",
    "fits well",
    "works well",
    "topic match",
    "content match",
    "good fit",
    "nice match",
    "matched",
    "aligns",
    "corresponds",
])


@dataclass
class ReasoningValidation:
    """Result of LLM reasoning validation."""
    is_valid: bool
    specific_references: int
    matched_keywords: List[str]
    warning_message: Optional[str] = None


def _extract_keywords(text: str, min_length: int = 3) -> Set[str]:
    """
    Extract significant keywords from text.

    Args:
        text: Input text to extract keywords from
        min_length: Minimum word length to consider

    Returns:
        Set of lowercase keywords
    """
    if not text:
        return set()

    # Remove punctuation and split into words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())

    # Filter by length and common stopwords
    stopwords = {
        'the', 'and', 'for', 'that', 'this', 'with', 'are', 'was', 'were',
        'has', 'have', 'had', 'been', 'being', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'into', 'from', 'about', 'which',
        'when', 'where', 'who', 'what', 'how', 'why', 'its', 'also', 'but',
        'not', 'than', 'then', 'these', 'those', 'some', 'any', 'all', 'each',
        'every', 'both', 'few', 'more', 'most', 'other', 'such', 'only', 'own',
        'same', 'very', 'just', 'because', 'before', 'after', 'during', 'while',
    }

    return {w for w in words if len(w) >= min_length and w not in stopwords}


def validate_llm_reasoning(
    reasoning: str,
    voiceover_text: str,
    video_text: Optional[str] = None,
    min_specific_references: int = 3
) -> ReasoningValidation:
    """
    Validate that LLM reasoning mentions specific keywords from voiceover or video.

    Flags low-quality reasoning that is generic (< 3 specific references).
    Logs warning when LLM provides generic reasoning.

    Args:
        reasoning: The LLM-generated reasoning string
        voiceover_text: Text from the voiceover segment
        video_text: Optional text from the matched video segment
        min_specific_references: Minimum keyword matches for valid reasoning (default: 3)

    Returns:
        ReasoningValidation with is_valid, specific_references count, and matched keywords
    """
    if not reasoning:
        warning = "Empty reasoning provided by LLM"
        logger.warning(warning)
        return ReasoningValidation(
            is_valid=False,
            specific_references=0,
            matched_keywords=[],
            warning_message=warning
        )

    reasoning_lower = reasoning.lower()

    # Check for generic phrases
    for phrase in GENERIC_REASONING_PHRASES:
        if phrase in reasoning_lower and len(reasoning_lower.strip()) < 30:
            # Very short reasoning with generic phrase
            warning = f"Generic LLM reasoning detected: '{reasoning[:50]}'"
            logger.warning(warning)
            return ReasoningValidation(
                is_valid=False,
                specific_references=0,
                matched_keywords=[],
                warning_message=warning
            )

    # Extract keywords from voiceover and video
    vo_keywords = _extract_keywords(voiceover_text)
    video_keywords = _extract_keywords(video_text) if video_text else set()
    all_source_keywords = vo_keywords | video_keywords

    # Extract keywords from reasoning
    reasoning_keywords = _extract_keywords(reasoning)

    # Find matches
    matched_keywords = list(reasoning_keywords & all_source_keywords)
    specific_references = len(matched_keywords)

    is_valid = specific_references >= min_specific_references

    if not is_valid:
        warning = (
            f"Low-quality LLM reasoning: only {specific_references} specific references "
            f"(need {min_specific_references}). Reasoning: '{reasoning[:50]}...'"
        )
        logger.warning(warning)
        return ReasoningValidation(
            is_valid=is_valid,
            specific_references=specific_references,
            matched_keywords=matched_keywords,
            warning_message=warning
        )

    return ReasoningValidation(
        is_valid=True,
        specific_references=specific_references,
        matched_keywords=matched_keywords,
        warning_message=None
    )


def select_negative_sample(
    candidates: List[Tuple[SRTSegment, float]],
    bottom_percentile: float = 0.25
) -> Optional[Tuple[SRTSegment, float]]:
    """
    Select a negative sample from the bottom percentile of candidates by similarity.

    Negative samples help LLMs calibrate confidence by showing what a poor match
    looks like. This improves match quality by giving the LLM a reference point
    for comparison.

    Args:
        candidates: List of (video_segment, similarity) tuples, sorted by similarity descending
        bottom_percentile: Percentile threshold for "bottom" candidates (default: 0.25 = bottom 25%)

    Returns:
        A single (segment, similarity) tuple from the bottom percentile, or None if
        insufficient candidates (need at least 4 to have meaningful bottom 25%)
    """
    if not candidates:
        logger.debug("select_negative_sample: no candidates provided")
        return None

    # Need at least 4 candidates to have a meaningful bottom 25%
    if len(candidates) < 4:
        logger.debug(f"select_negative_sample: insufficient candidates ({len(candidates)}), need >= 4")
        return None

    # Calculate the cutoff index for bottom percentile
    cutoff_index = int(len(candidates) * (1 - bottom_percentile))

    # Get bottom percentile candidates
    bottom_candidates = candidates[cutoff_index:]

    if not bottom_candidates:
        logger.debug("select_negative_sample: no bottom candidates after cutoff")
        return None

    # Randomly select one from the bottom candidates
    # Random selection prevents predictable patterns and adds variety
    selected = random.choice(bottom_candidates)

    logger.debug(
        f"select_negative_sample: selected negative from bottom {int(bottom_percentile*100)}% "
        f"(index {cutoff_index}-{len(candidates)-1}), similarity={selected[1]:.3f}"
    )

    return selected


def format_negative_sample_for_prompt(
    negative: Tuple[SRTSegment, float],
    index_label: str = "UNLIKELY"
) -> str:
    """
    Format a negative sample for inclusion in an LLM prompt.

    Args:
        negative: Tuple of (segment, similarity) for the negative sample
        index_label: Label to use for the negative sample (default: "UNLIKELY")

    Returns:
        Formatted string for inclusion in prompt
    """
    seg, sim = negative
    source_name = Path(seg.source_file).stem[:30] if seg.source_file else "unknown"
    text_preview = seg.text[:60].replace('"', "'") if seg.text else ""
    if len(seg.text) > 60:
        text_preview += "..."

    return f"  {index_label}. [{source_name}] \"{text_preview}\" (unlikely match - for comparison)"


class LLMProvider(ABC):
    """Base class for LLM providers"""

    @abstractmethod
    def match_batch(
        self,
        items: List[Tuple[str, List[Tuple[SRTSegment, float]]]],
        context: Optional[str] = None,
        negative_rules: Optional[List[str]] = None,
        negative_samples: Optional[List[Optional[Tuple[SRTSegment, float]]]] = None
    ) -> List[Tuple[int, float, str]]:
        """
        Batch match voiceover segments to candidates.

        Args:
            items: List of (voiceover_text, candidates) tuples
            context: Optional context string for the batch
            negative_rules: Optional list of negative matching rules
            negative_samples: Optional list of negative samples (one per item), used for
                              calibrating confidence by showing what a poor match looks like

        Returns:
            List of (selected_idx, confidence, reasoning) tuples
        """
        pass


class GeminiMatcher(LLMProvider):
    """Gemini Flash for matching"""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        from src.llm_client import create_client
        self.client = create_client("gemini", api_key=api_key, model=model)

    def match_batch(
        self,
        items: List[Tuple[str, List[Tuple[SRTSegment, float]]]],
        context: Optional[str] = None,
        negative_rules: Optional[List[str]] = None,
        negative_samples: Optional[List[Optional[Tuple[SRTSegment, float]]]] = None
    ) -> List[Tuple[int, float, str]]:
        from src.llm_client import LLMRequest, ResponseFormat

        # Build batch prompt
        batch_sections = []
        for i, (vo_text, candidates) in enumerate(items):
            # Escape quotes in text to avoid JSON issues
            vo_text_clean = vo_text.replace('"', "'")
            candidates_text = "\n".join([
                f"  {j+1}. [{Path(seg.source_file).stem[:30]}] \"{seg.text[:60].replace(chr(34), chr(39))}{'...' if len(seg.text) > 60 else ''}\""
                for j, (seg, sim) in enumerate(candidates[:5])
            ])

            # Add negative sample if provided for this item
            negative_sample_text = ""
            if negative_samples and i < len(negative_samples) and negative_samples[i]:
                negative_sample_text = "\n" + format_negative_sample_for_prompt(negative_samples[i])

            batch_sections.append(f"VOICEOVER {i+1}: \"{vo_text_clean[:100]}\"\nCANDIDATES:\n{candidates_text}{negative_sample_text}")

        context_str = f"\nCONTEXT: {context}" if context else ""

        negative_str = ""
        if negative_rules:
            negative_str = "\n\nAVOID:\n" + "\n".join(f"- {rule}" for rule in negative_rules)

        # Include instruction about negative sample if any are present
        negative_instruction = ""
        if negative_samples and any(ns is not None for ns in negative_samples):
            negative_instruction = "\nNote: Candidates marked 'unlikely match' are poor matches shown for comparison - do NOT select them. Use them to calibrate your confidence scoring."

        prompt = f"""Match each voiceover to its best video candidate based on semantic meaning and topic alignment.
{context_str}{negative_str}{negative_instruction}

{chr(10).join(batch_sections)}

Respond with ONLY a valid JSON array, no other text. Use simple reasons without special characters:
[{{"voiceover": 1, "selected": 1, "confidence": 0.85, "reason": "topic match"}}]"""

        logger.info(f"    GeminiMatcher: sending request (timeout=120s)...")

        # Call unified LLM client (handles retry and parsing)
        request = LLMRequest(
            prompt=prompt,
            response_format=ResponseFormat.JSON_ARRAY,
            cache_key_prefix="matching",
            timeout=120
        )

        try:
            response = self.client.generate(request)
            logger.info(f"    GeminiMatcher: received response")

            # Process results
            if response.parsed_data and isinstance(response.parsed_data, list):
                results_list = response.parsed_data
                outputs = []
                for i, (vo_text, candidates) in enumerate(items):
                    result = next((r for r in results_list if r.get('voiceover') == i + 1), None)
                    if result:
                        selected_idx = result.get('selected', 1) - 1
                        # Clamp to valid range
                        selected_idx = max(0, min(selected_idx, len(candidates) - 1))
                        confidence = result.get('confidence', 0.7)
                        # Clamp confidence to valid range
                        confidence = max(0.0, min(1.0, float(confidence)))
                        outputs.append((
                            selected_idx,
                            confidence,
                            str(result.get('reason', 'matched'))[:50]
                        ))
                    else:
                        # Fallback for missing voiceover entry
                        outputs.append((0, candidates[0][1] if candidates else 0.5, "parse fallback"))

                return outputs
            else:
                logger.warning(f"Gemini: Could not parse response, using embedding fallback")
                raise ValueError("JSON parsing failed")

        except Exception as e:
            logger.warning(f"Gemini batch error: {e}")
            raise

        return [(0, 0.5, "error fallback") for _ in items]


class ClaudeMatcher(LLMProvider):
    """Claude Haiku for matching (secondary/ambiguous)"""

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        from src.llm_client import create_client
        self.client = create_client("anthropic", api_key=api_key, model=model)

    def match_batch(
        self,
        items: List[Tuple[str, List[Tuple[SRTSegment, float]]]],
        context: Optional[str] = None,
        negative_rules: Optional[List[str]] = None,
        negative_samples: Optional[List[Optional[Tuple[SRTSegment, float]]]] = None
    ) -> List[Tuple[int, float, str]]:
        from src.llm_client import LLMRequest, ResponseFormat

        batch_sections = []
        for i, (vo_text, candidates) in enumerate(items):
            # Escape quotes in text to avoid JSON issues
            vo_text_clean = vo_text.replace('"', "'")[:100]
            candidates_text = "\n".join([
                f"  {j+1}. [{Path(seg.source_file).stem[:30]}] \"{seg.text[:60].replace(chr(34), chr(39))}{'...' if len(seg.text) > 60 else ''}\""
                for j, (seg, sim) in enumerate(candidates[:5])
            ])

            # Add negative sample if provided for this item
            negative_sample_text = ""
            if negative_samples and i < len(negative_samples) and negative_samples[i]:
                negative_sample_text = "\n" + format_negative_sample_for_prompt(negative_samples[i])

            batch_sections.append(f"VOICEOVER {i+1}: \"{vo_text_clean}\"\nCANDIDATES:\n{candidates_text}{negative_sample_text}")

        context_str = f"\nCONTEXT: {context}" if context else ""

        negative_str = ""
        if negative_rules:
            negative_str = "\n\nAVOID:\n" + "\n".join(f"- {rule}" for rule in negative_rules)

        # Include instruction about negative sample if any are present
        negative_instruction = ""
        if negative_samples and any(ns is not None for ns in negative_samples):
            negative_instruction = "\nNote: Candidates marked 'unlikely match' are poor matches shown for comparison - do NOT select them. Use them to calibrate your confidence scoring."

        prompt = f"""Match each voiceover to its best video candidate. Consider semantic meaning, visual relevance, and topic alignment.
{context_str}{negative_str}{negative_instruction}

{chr(10).join(batch_sections)}

Respond with ONLY a valid JSON array, no other text. Use simple reasons without special characters:
[{{"voiceover": 1, "selected": 1, "confidence": 0.85, "reason": "topic match"}}]"""

        # Call unified LLM client (handles retry and parsing)
        request = LLMRequest(
            prompt=prompt,
            response_format=ResponseFormat.JSON_ARRAY,
            cache_key_prefix="matching",
            max_tokens=1500,
            timeout=120
        )

        try:
            response = self.client.generate(request)

            # Process results
            if response.parsed_data and isinstance(response.parsed_data, list):
                results_list = response.parsed_data
                outputs = []
                for i, (vo_text, candidates) in enumerate(items):
                    result = next((r for r in results_list if r.get('voiceover') == i + 1), None)
                    if result:
                        selected_idx = result.get('selected', 1) - 1
                        # Clamp to valid range
                        selected_idx = max(0, min(selected_idx, len(candidates) - 1))
                        confidence = result.get('confidence', 0.7)
                        # Clamp confidence to valid range
                        confidence = max(0.0, min(1.0, float(confidence)))
                        outputs.append((
                            selected_idx,
                            confidence,
                            str(result.get('reason', 'matched'))[:50]
                        ))
                    else:
                        outputs.append((0, candidates[0][1] if candidates else 0.5, "parse fallback"))

                return outputs
            else:
                logger.warning(f"Claude: Could not parse response, using embedding fallback")
                raise ValueError("JSON parsing failed")

        except Exception as e:
            logger.warning(f"Claude batch error: {e}")
            raise

        return [(0, 0.5, "error fallback") for _ in items]


class LocalLLMMatcher(LLMProvider):
    """Local LLM (Ollama) for finishing touches"""

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        from src.llm_client import create_client
        self.client = create_client("ollama", model=model, host=host)

    def match_batch(
        self,
        items: List[Tuple[str, List[Tuple[SRTSegment, float]]]],
        context: Optional[str] = None,
        negative_rules: Optional[List[str]] = None,
        negative_samples: Optional[List[Optional[Tuple[SRTSegment, float]]]] = None
    ) -> List[Tuple[int, float, str]]:
        from src.llm_client import LLMRequest, ResponseFormat

        outputs = []

        # Process items one at a time (Ollama works better this way)
        for idx, (vo_text, candidates) in enumerate(items):
            # Escape quotes to avoid JSON issues
            vo_text_clean = vo_text.replace('"', "'")[:100]
            candidates_text = "\n".join([
                f"{j+1}. \"{seg.text[:80].replace(chr(34), chr(39))}\""
                for j, (seg, sim) in enumerate(candidates[:5])
            ])

            # Add negative sample if provided
            negative_sample_text = ""
            if negative_samples and idx < len(negative_samples) and negative_samples[idx]:
                neg_seg, neg_sim = negative_samples[idx]
                neg_text = neg_seg.text[:80].replace('"', "'") if neg_seg.text else ""
                negative_sample_text = f"\nUNLIKELY. \"{neg_text}\" (poor match - for comparison)"

            # Add instruction about negative sample
            negative_instruction = ""
            if negative_sample_text:
                negative_instruction = " Do NOT select UNLIKELY - it shows what a poor match looks like."

            prompt = f"""Match this voiceover to the best candidate:{negative_instruction}

VOICEOVER: "{vo_text_clean}"

CANDIDATES:
{candidates_text}{negative_sample_text}

Respond with ONLY valid JSON, no other text: {{"selected": 1, "confidence": 0.85, "reason": "topic match"}}"""

            try:
                # Call unified LLM client
                request = LLMRequest(
                    prompt=prompt,
                    response_format=ResponseFormat.JSON,
                    cache_key_prefix="matching",
                    timeout=60
                )

                response = self.client.generate(request)

                # Process result
                if response.parsed_data and isinstance(response.parsed_data, dict):
                    data = response.parsed_data
                    selected_idx = data.get('selected', 1) - 1
                    selected_idx = max(0, min(selected_idx, len(candidates) - 1))
                    confidence = max(0.0, min(1.0, float(data.get('confidence', 0.5))))
                    outputs.append((
                        selected_idx,
                        confidence,
                        str(data.get('reason', 'local match'))[:50]
                    ))
                else:
                    # Fallback
                    outputs.append((0, candidates[0][1] if candidates else 0.5, "local fallback"))

            except Exception as e:
                logger.debug(f"Local LLM error: {e}")
                # Fallback
                outputs.append((0, candidates[0][1] if candidates else 0.5, "local fallback"))

        return outputs
