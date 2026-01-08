"""
LLM provider implementations for matching.

Migrated from matching.py lines 228-484.
All providers use unified src/llm_client/ (Rule 9).

Providers:
- LLMProvider: Abstract base class
- GeminiMatcher: Google Gemini Flash (primary provider)
- ClaudeMatcher: Anthropic Claude Haiku (secondary/ambiguous)
- LocalLLMMatcher: Ollama (local LLM, one-at-a-time processing)
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

from ..utils import SRTSegment

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Base class for LLM providers"""

    @abstractmethod
    def match_batch(
        self,
        items: List[Tuple[str, List[Tuple[SRTSegment, float]]]],
        context: Optional[str] = None,
        negative_rules: Optional[List[str]] = None
    ) -> List[Tuple[int, float, str]]:
        """
        Batch match voiceover segments to candidates.
        Returns list of (selected_idx, confidence, reasoning)
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
        negative_rules: Optional[List[str]] = None
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
            batch_sections.append(f"VOICEOVER {i+1}: \"{vo_text_clean[:100]}\"\nCANDIDATES:\n{candidates_text}")

        context_str = f"\nCONTEXT: {context}" if context else ""

        negative_str = ""
        if negative_rules:
            negative_str = "\n\nAVOID:\n" + "\n".join(f"- {rule}" for rule in negative_rules)

        prompt = f"""Match each voiceover to its best video candidate based on semantic meaning and topic alignment.
{context_str}{negative_str}

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
        negative_rules: Optional[List[str]] = None
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
            batch_sections.append(f"VOICEOVER {i+1}: \"{vo_text_clean}\"\nCANDIDATES:\n{candidates_text}")

        context_str = f"\nCONTEXT: {context}" if context else ""

        negative_str = ""
        if negative_rules:
            negative_str = "\n\nAVOID:\n" + "\n".join(f"- {rule}" for rule in negative_rules)

        prompt = f"""Match each voiceover to its best video candidate. Consider semantic meaning, visual relevance, and topic alignment.
{context_str}{negative_str}

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
        negative_rules: Optional[List[str]] = None
    ) -> List[Tuple[int, float, str]]:
        from src.llm_client import LLMRequest, ResponseFormat

        outputs = []

        # Process items one at a time (Ollama works better this way)
        for vo_text, candidates in items:
            # Escape quotes to avoid JSON issues
            vo_text_clean = vo_text.replace('"', "'")[:100]
            candidates_text = "\n".join([
                f"{j+1}. \"{seg.text[:80].replace(chr(34), chr(39))}\""
                for j, (seg, sim) in enumerate(candidates[:5])
            ])

            prompt = f"""Match this voiceover to the best candidate:

VOICEOVER: "{vo_text_clean}"

CANDIDATES:
{candidates_text}

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
