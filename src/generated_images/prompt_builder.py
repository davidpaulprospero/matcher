"""Prompt builder for generated subtitle-driven stills."""

from __future__ import annotations

import re
from typing import List


def _dedupe(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


class GeneratedImagePromptBuilder:
    """Build prompts from grouped subtitle text and config knobs."""

    def __init__(self, config: object):
        self.config = config

    def build(self, subtitle_text: str, topic_context: str = "") -> str:
        cleaned_text = self._clean_text(subtitle_text)
        if not cleaned_text:
            cleaned_text = "Detailed scene matching the voiceover context."

        parts: List[str] = []

        prompt_prefix = getattr(self.config, 'prompt_prefix', '').strip()
        if prompt_prefix:
            parts.append(prompt_prefix)

        if getattr(self.config, 'include_topic_context', True) and topic_context.strip():
            parts.append(f"Topic context: {topic_context.strip()}.")

        parts.append(f"Scene: {cleaned_text}")

        if getattr(self.config, 'use_prompt_enhancement', True):
            provider_keywords = list(getattr(self.config, 'provider_keywords', []))
            quality_keywords = list(getattr(self.config, 'quality_keywords', []))
            enhancement = _dedupe(provider_keywords + quality_keywords)
            if enhancement:
                parts.append(", ".join(enhancement))

        quality = getattr(self.config, 'quality', '').strip()
        if quality and quality.lower() != 'standard':
            parts.append(f"Quality target: {quality}")

        prompt_suffix = getattr(self.config, 'prompt_suffix', '').strip()
        if prompt_suffix:
            parts.append(prompt_suffix)

        return " ".join(parts).strip()

    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        return cleaned
