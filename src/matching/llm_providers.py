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
- build_cot_prompt: Build chain-of-thought structured prompt
- parse_cot_reasoning: Parse structured CoT response into components
"""

import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

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


@dataclass
class ExplanationValidation:
    """Result of LLM explanation confidence validation.

    Cross-checks that keywords mentioned in the LLM's explanation
    actually appear in the source content (voiceover/video text).

    Used to detect and penalize "hallucinated" explanations where
    the LLM references content that doesn't exist.
    """
    is_valid: bool
    verification_ratio: float  # Ratio of verifiable keywords (0.0-1.0)
    explanation_keywords: List[str]  # Keywords extracted from explanation
    verified_keywords: List[str]  # Keywords found in source content
    unverified_keywords: List[str]  # Keywords NOT found in source content
    confidence_penalty: float  # Amount to reduce confidence (0.0 or 0.1)
    warning_message: Optional[str] = None


# Chain-of-thought scoring rubric weights
COT_RUBRIC_WEIGHTS = {
    'visual_relevance': 0.30,  # How well video visuals match voiceover content
    'topic_match': 0.40,       # Alignment of video topic with voiceover theme
    'keyword_overlap': 0.20,   # Specific keywords shared between voiceover and video
    'flow': 0.10               # How well the video maintains narrative flow
}


@dataclass
class CotReasoning:
    """Parsed chain-of-thought reasoning from LLM response.

    The CoT structure guides LLMs through a 4-step reasoning process:
    1. Identify key voiceover themes
    2. List matching elements in video
    3. Evaluate fit using the rubric
    4. Compute final score

    This produces more reliable and explainable match decisions.
    """
    voiceover_themes: List[str] = field(default_factory=list)
    video_elements: List[str] = field(default_factory=list)
    rubric_scores: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0
    reasoning_text: str = ""

    @property
    def is_complete(self) -> bool:
        """Check if all reasoning components are present."""
        return (
            len(self.voiceover_themes) > 0 and
            len(self.video_elements) > 0 and
            len(self.rubric_scores) > 0 and
            self.final_score > 0
        )

    def compute_weighted_score(self) -> float:
        """Compute final score from rubric scores using weights."""
        total = 0.0
        weight_sum = 0.0
        for key, weight in COT_RUBRIC_WEIGHTS.items():
            if key in self.rubric_scores:
                total += self.rubric_scores[key] * weight
                weight_sum += weight
        if weight_sum > 0:
            return total / weight_sum * 1.0  # Normalize to 0-1
        return self.final_score


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


# Confidence penalty when explanation verification fails
EXPLANATION_VERIFICATION_THRESHOLD = 0.5  # Minimum ratio of verifiable keywords
EXPLANATION_CONFIDENCE_PENALTY = 0.1  # Penalty when below threshold


def validate_explanation_confidence(
    explanation: str,
    voiceover_text: str,
    video_text: Optional[str] = None,
    verification_threshold: float = EXPLANATION_VERIFICATION_THRESHOLD,
    confidence_penalty: float = EXPLANATION_CONFIDENCE_PENALTY
) -> ExplanationValidation:
    """
    Validate that LLM explanation keywords appear in actual voiceover/video text.

    Cross-checks keywords mentioned in the LLM's explanation against the source
    content to detect "hallucinated" explanations that reference non-existent
    content. When less than 50% of explanation keywords can be verified in the
    source content, confidence is downgraded by 0.1.

    Args:
        explanation: The LLM-generated explanation/reasoning string
        voiceover_text: Text from the voiceover segment
        video_text: Optional text from the matched video segment (transcript)
        verification_threshold: Minimum ratio of verifiable keywords (default: 0.5)
        confidence_penalty: Amount to reduce confidence when below threshold (default: 0.1)

    Returns:
        ExplanationValidation with verification results and confidence penalty

    Example:
        >>> result = validate_explanation_confidence(
        ...     explanation="The sunset imagery matches the peaceful narration",
        ...     voiceover_text="A peaceful evening by the ocean",
        ...     video_text="sunset over the water with waves"
        ... )
        >>> result.is_valid
        True
        >>> result.verification_ratio
        0.75  # "sunset", "peaceful" found; "imagery" not found
    """
    if not explanation:
        warning = "Empty explanation provided by LLM"
        logger.warning(warning)
        return ExplanationValidation(
            is_valid=False,
            verification_ratio=0.0,
            explanation_keywords=[],
            verified_keywords=[],
            unverified_keywords=[],
            confidence_penalty=confidence_penalty,
            warning_message=warning
        )

    # Extract keywords from explanation
    explanation_keywords = _extract_keywords(explanation)

    if not explanation_keywords:
        # No meaningful keywords to verify
        logger.debug("validate_explanation_confidence: No keywords extracted from explanation")
        return ExplanationValidation(
            is_valid=True,
            verification_ratio=1.0,  # Nothing to verify, assume valid
            explanation_keywords=[],
            verified_keywords=[],
            unverified_keywords=[],
            confidence_penalty=0.0,
            warning_message=None
        )

    # Extract keywords from source content (voiceover and video)
    vo_keywords = _extract_keywords(voiceover_text)
    video_keywords = _extract_keywords(video_text) if video_text else set()
    source_keywords = vo_keywords | video_keywords

    # Find which explanation keywords are verifiable in source
    verified_keywords = explanation_keywords & source_keywords
    unverified_keywords = explanation_keywords - source_keywords

    # Calculate verification ratio
    verification_ratio = len(verified_keywords) / len(explanation_keywords)

    # Determine if validation passes
    is_valid = verification_ratio >= verification_threshold
    applied_penalty = 0.0 if is_valid else confidence_penalty

    # Log warning when explanation references non-existent content
    warning_message = None
    if not is_valid:
        unverified_list = sorted(unverified_keywords)[:5]  # Limit to 5 for readability
        warning_message = (
            f"LLM explanation references non-existent content: "
            f"only {len(verified_keywords)}/{len(explanation_keywords)} keywords verifiable "
            f"({verification_ratio:.1%}). Unverified: {unverified_list}. "
            f"Applying -{applied_penalty} confidence penalty."
        )
        logger.warning(warning_message)
    else:
        logger.debug(
            f"validate_explanation_confidence: {len(verified_keywords)}/{len(explanation_keywords)} "
            f"keywords verified ({verification_ratio:.1%})"
        )

    return ExplanationValidation(
        is_valid=is_valid,
        verification_ratio=verification_ratio,
        explanation_keywords=sorted(explanation_keywords),
        verified_keywords=sorted(verified_keywords),
        unverified_keywords=sorted(unverified_keywords),
        confidence_penalty=applied_penalty,
        warning_message=warning_message
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


def build_cot_prompt(
    voiceover_text: str,
    candidates: List[Tuple[SRTSegment, float]],
    negative_sample: Optional[Tuple[SRTSegment, float]] = None,
    context: Optional[str] = None,
    negative_rules: Optional[List[str]] = None
) -> str:
    """
    Build a chain-of-thought structured prompt for LLM matching.

    The CoT prompt guides the LLM through 4 reasoning steps:
    1. Identify key voiceover themes
    2. List matching elements in video
    3. Evaluate fit using explicit rubric
    4. Compute final score

    Rubric weights:
    - Visual relevance: 30%
    - Topic match: 40%
    - Keyword overlap: 20%
    - Flow/pacing: 10%

    Args:
        voiceover_text: The voiceover text to match
        candidates: List of (video_segment, similarity) tuples
        negative_sample: Optional negative sample for calibration
        context: Optional context string
        negative_rules: Optional list of things to avoid

    Returns:
        Formatted CoT prompt string
    """
    vo_text_clean = voiceover_text.replace('"', "'")[:100]

    # Format candidates
    candidates_text = "\n".join([
        f"  {j+1}. [{Path(seg.source_file).stem[:30] if seg.source_file else 'unknown'}] "
        f"\"{seg.text[:60].replace(chr(34), chr(39))}{'...' if len(seg.text) > 60 else ''}\""
        for j, (seg, sim) in enumerate(candidates[:5])
    ])

    # Add negative sample if provided
    negative_sample_text = ""
    if negative_sample:
        negative_sample_text = "\n" + format_negative_sample_for_prompt(negative_sample)

    # Build context and rules sections
    context_str = f"\nCONTEXT: {context}" if context else ""
    negative_str = ""
    if negative_rules:
        negative_str = "\n\nAVOID:\n" + "\n".join(f"- {rule}" for rule in negative_rules)

    negative_instruction = ""
    if negative_sample:
        negative_instruction = "\nNote: Candidates marked 'unlikely match' are poor matches - do NOT select them."

    # Chain-of-thought structured prompt
    prompt = f"""Match the voiceover to the best video candidate using structured reasoning.
{context_str}{negative_str}{negative_instruction}

VOICEOVER: "{vo_text_clean}"

CANDIDATES:
{candidates_text}{negative_sample_text}

## SCORING RUBRIC (use these weights):
- Visual Relevance (30%): How well video visuals match voiceover content
- Topic Match (40%): Alignment of video topic with voiceover theme
- Keyword Overlap (20%): Specific keywords shared between voiceover and video
- Flow (10%): How well the video maintains narrative pacing

## REASONING STEPS (follow this structure):

STEP 1 - VOICEOVER THEMES: List 2-3 key themes from the voiceover
STEP 2 - VIDEO ELEMENTS: For best candidate, list matching visual elements
STEP 3 - RUBRIC EVALUATION: Score each rubric dimension (0.0-1.0)
STEP 4 - FINAL SCORE: Compute weighted average

## RESPONSE FORMAT (JSON only, no other text):
{{
  "selected": 1,
  "voiceover_themes": ["theme1", "theme2"],
  "video_elements": ["element1", "element2"],
  "rubric_scores": {{"visual_relevance": 0.8, "topic_match": 0.9, "keyword_overlap": 0.7, "flow": 0.8}},
  "confidence": 0.85,
  "reason": "brief summary"
}}"""

    return prompt


def parse_cot_reasoning(response_data: dict) -> CotReasoning:
    """
    Parse structured CoT response from LLM into CotReasoning object.

    Extracts:
    - voiceover_themes: List of identified themes
    - video_elements: List of matching video elements
    - rubric_scores: Dict of rubric dimension scores
    - final_score: Computed confidence score
    - reasoning_text: Brief reason text

    Args:
        response_data: Parsed JSON response from LLM

    Returns:
        CotReasoning object with parsed components
    """
    if not response_data or not isinstance(response_data, dict):
        logger.warning("parse_cot_reasoning: Invalid response data")
        return CotReasoning()

    # Extract themes
    voiceover_themes = response_data.get('voiceover_themes', [])
    if isinstance(voiceover_themes, str):
        voiceover_themes = [voiceover_themes]
    elif not isinstance(voiceover_themes, list):
        voiceover_themes = []

    # Extract video elements
    video_elements = response_data.get('video_elements', [])
    if isinstance(video_elements, str):
        video_elements = [video_elements]
    elif not isinstance(video_elements, list):
        video_elements = []

    # Extract rubric scores
    rubric_scores = response_data.get('rubric_scores', {})
    if not isinstance(rubric_scores, dict):
        rubric_scores = {}

    # Validate and normalize rubric scores
    normalized_scores = {}
    for key in COT_RUBRIC_WEIGHTS:
        if key in rubric_scores:
            try:
                score = float(rubric_scores[key])
                normalized_scores[key] = max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                logger.debug(f"parse_cot_reasoning: Invalid score for {key}")

    # Get confidence/final score
    final_score = response_data.get('confidence', 0.0)
    try:
        final_score = max(0.0, min(1.0, float(final_score)))
    except (ValueError, TypeError):
        final_score = 0.0

    # Get reasoning text
    reasoning_text = str(response_data.get('reason', ''))[:100]

    cot = CotReasoning(
        voiceover_themes=voiceover_themes,
        video_elements=video_elements,
        rubric_scores=normalized_scores,
        final_score=final_score,
        reasoning_text=reasoning_text
    )

    # Log if incomplete
    if not cot.is_complete:
        logger.debug(
            f"parse_cot_reasoning: Incomplete reasoning - "
            f"themes={len(voiceover_themes)}, elements={len(video_elements)}, "
            f"rubric={len(normalized_scores)}, score={final_score}"
        )

    return cot


def build_cot_batch_prompt(
    items: List[Tuple[str, List[Tuple[SRTSegment, float]]]],
    context: Optional[str] = None,
    negative_rules: Optional[List[str]] = None,
    negative_samples: Optional[List[Optional[Tuple[SRTSegment, float]]]] = None
) -> str:
    """
    Build a batched chain-of-thought prompt for multiple voiceover-candidate pairs.

    Args:
        items: List of (voiceover_text, candidates) tuples
        context: Optional context string for the batch
        negative_rules: Optional list of negative matching rules
        negative_samples: Optional list of negative samples (one per item)

    Returns:
        Formatted batched CoT prompt string
    """
    batch_sections = []
    for i, (vo_text, candidates) in enumerate(items):
        vo_text_clean = vo_text.replace('"', "'")[:100]
        candidates_text = "\n".join([
            f"  {j+1}. [{Path(seg.source_file).stem[:30] if seg.source_file else 'unknown'}] "
            f"\"{seg.text[:60].replace(chr(34), chr(39))}{'...' if len(seg.text) > 60 else ''}\""
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

    negative_instruction = ""
    if negative_samples and any(ns is not None for ns in negative_samples):
        negative_instruction = "\nNote: Candidates marked 'unlikely match' are poor matches - do NOT select them."

    prompt = f"""Match each voiceover to its best video candidate using structured reasoning.
{context_str}{negative_str}{negative_instruction}

{chr(10).join(batch_sections)}

## SCORING RUBRIC (use these weights for each match):
- Visual Relevance (30%): How well video visuals match voiceover content
- Topic Match (40%): Alignment of video topic with voiceover theme
- Keyword Overlap (20%): Specific keywords shared between voiceover and video
- Flow (10%): How well the video maintains narrative pacing

## REASONING STEPS (for each voiceover):
1. Identify 2-3 key themes from voiceover
2. List matching visual elements in best candidate
3. Score each rubric dimension (0.0-1.0)
4. Compute weighted average for final confidence

## RESPONSE FORMAT (JSON array only, no other text):
[{{
  "voiceover": 1,
  "selected": 1,
  "voiceover_themes": ["theme1", "theme2"],
  "video_elements": ["element1", "element2"],
  "rubric_scores": {{"visual_relevance": 0.8, "topic_match": 0.9, "keyword_overlap": 0.7, "flow": 0.8}},
  "confidence": 0.85,
  "reason": "brief summary"
}}]"""

    return prompt


class LLMProvider(ABC):
    """Base class for LLM providers"""

    @abstractmethod
    def match_batch(
        self,
        items: List[Tuple[str, List[Tuple[SRTSegment, float]]]],
        context: Optional[str] = None,
        negative_rules: Optional[List[str]] = None,
        negative_samples: Optional[List[Optional[Tuple[SRTSegment, float]]]] = None,
        use_cot: bool = False
    ) -> List[Tuple[int, float, str, Optional[CotReasoning]]]:
        """
        Batch match voiceover segments to candidates.

        Args:
            items: List of (voiceover_text, candidates) tuples
            context: Optional context string for the batch
            negative_rules: Optional list of negative matching rules
            negative_samples: Optional list of negative samples (one per item), used for
                              calibrating confidence by showing what a poor match looks like
            use_cot: Whether to use chain-of-thought prompting (default: False)

        Returns:
            List of (selected_idx, confidence, reasoning, cot_reasoning) tuples.
            cot_reasoning is populated when use_cot=True, None otherwise.
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
        negative_samples: Optional[List[Optional[Tuple[SRTSegment, float]]]] = None,
        use_cot: bool = False
    ) -> List[Tuple[int, float, str, Optional[CotReasoning]]]:
        from src.llm_client import LLMRequest, ResponseFormat

        # Use CoT prompt when enabled
        if use_cot:
            prompt = build_cot_batch_prompt(items, context, negative_rules, negative_samples)
            logger.info(f"    GeminiMatcher: using chain-of-thought prompt (timeout=120s)...")
        else:
            # Build standard batch prompt
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
            cache_key_prefix="matching_cot" if use_cot else "matching",
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

                        # Parse CoT reasoning if CoT is enabled
                        cot_reasoning = None
                        if use_cot:
                            cot_reasoning = parse_cot_reasoning(result)
                            # Use weighted score if CoT rubric is complete
                            if cot_reasoning.is_complete:
                                weighted_score = cot_reasoning.compute_weighted_score()
                                # Blend with LLM confidence (70% weighted, 30% LLM)
                                confidence = 0.7 * weighted_score + 0.3 * confidence
                                logger.debug(
                                    f"    CoT: weighted={weighted_score:.3f}, blended={confidence:.3f}"
                                )

                        outputs.append((
                            selected_idx,
                            confidence,
                            str(result.get('reason', 'matched'))[:50],
                            cot_reasoning
                        ))
                    else:
                        # Fallback for missing voiceover entry
                        outputs.append((0, candidates[0][1] if candidates else 0.5, "parse fallback", None))

                return outputs
            else:
                logger.warning(f"Gemini: Could not parse response, using embedding fallback")
                raise ValueError("JSON parsing failed")

        except Exception as e:
            logger.warning(f"Gemini batch error: {e}")
            raise

        return [(0, 0.5, "error fallback", None) for _ in items]


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
        negative_samples: Optional[List[Optional[Tuple[SRTSegment, float]]]] = None,
        use_cot: bool = False
    ) -> List[Tuple[int, float, str, Optional[CotReasoning]]]:
        from src.llm_client import LLMRequest, ResponseFormat

        # Use CoT prompt when enabled
        if use_cot:
            prompt = build_cot_batch_prompt(items, context, negative_rules, negative_samples)
        else:
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
            cache_key_prefix="matching_cot" if use_cot else "matching",
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

                        # Parse CoT reasoning if CoT is enabled
                        cot_reasoning = None
                        if use_cot:
                            cot_reasoning = parse_cot_reasoning(result)
                            if cot_reasoning.is_complete:
                                weighted_score = cot_reasoning.compute_weighted_score()
                                confidence = 0.7 * weighted_score + 0.3 * confidence

                        outputs.append((
                            selected_idx,
                            confidence,
                            str(result.get('reason', 'matched'))[:50],
                            cot_reasoning
                        ))
                    else:
                        outputs.append((0, candidates[0][1] if candidates else 0.5, "parse fallback", None))

                return outputs
            else:
                logger.warning(f"Claude: Could not parse response, using embedding fallback")
                raise ValueError("JSON parsing failed")

        except Exception as e:
            logger.warning(f"Claude batch error: {e}")
            raise

        return [(0, 0.5, "error fallback", None) for _ in items]


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
        negative_samples: Optional[List[Optional[Tuple[SRTSegment, float]]]] = None,
        use_cot: bool = False
    ) -> List[Tuple[int, float, str, Optional[CotReasoning]]]:
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

            # Use CoT prompt when enabled (simplified for local LLM)
            if use_cot:
                prompt = f"""Match this voiceover to the best candidate using structured reasoning:{negative_instruction}

VOICEOVER: "{vo_text_clean}"

CANDIDATES:
{candidates_text}{negative_sample_text}

RUBRIC: visual_relevance (30%), topic_match (40%), keyword_overlap (20%), flow (10%)

Respond with ONLY valid JSON:
{{"selected": 1, "voiceover_themes": ["theme1"], "video_elements": ["element1"], "rubric_scores": {{"visual_relevance": 0.8, "topic_match": 0.9, "keyword_overlap": 0.7, "flow": 0.8}}, "confidence": 0.85, "reason": "topic match"}}"""
            else:
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
                    cache_key_prefix="matching_cot" if use_cot else "matching",
                    timeout=60
                )

                response = self.client.generate(request)

                # Process result
                if response.parsed_data and isinstance(response.parsed_data, dict):
                    data = response.parsed_data
                    selected_idx = data.get('selected', 1) - 1
                    selected_idx = max(0, min(selected_idx, len(candidates) - 1))
                    confidence = max(0.0, min(1.0, float(data.get('confidence', 0.5))))

                    # Parse CoT reasoning if CoT is enabled
                    cot_reasoning = None
                    if use_cot:
                        cot_reasoning = parse_cot_reasoning(data)
                        if cot_reasoning.is_complete:
                            weighted_score = cot_reasoning.compute_weighted_score()
                            confidence = 0.7 * weighted_score + 0.3 * confidence

                    outputs.append((
                        selected_idx,
                        confidence,
                        str(data.get('reason', 'local match'))[:50],
                        cot_reasoning
                    ))
                else:
                    # Fallback
                    outputs.append((0, candidates[0][1] if candidates else 0.5, "local fallback", None))

            except Exception as e:
                logger.debug(f"Local LLM error: {e}")
                # Fallback
                outputs.append((0, candidates[0][1] if candidates else 0.5, "local fallback", None))

        return outputs
