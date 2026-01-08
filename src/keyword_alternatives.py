"""
Unified LLM-based keyword alternative generation.

Consolidates duplicate logic from:
- src/keyword_remix.py (lines 975-1056): Post-download keyword remixing
- src/downloader/keyword_remix.py (lines 48-119): Download search optimization

Created Jan 6, 2026 as part of keyword_remix naming collision fix.
Uses unified src.llm_client (Rule 9 compliant).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)


class KeywordAlternativeGenerator:
    """Generates alternative keywords using LLM when searches fail.

    Supports two use cases:
    1. Single keyword generation (for download search optimization)
    2. Multiple keyword generation (for post-download remix filtering)

    Uses unified src.llm_client (Rule 9 compliant).
    """

    # Prompt for single keyword alternative (download search optimization)
    SINGLE_KEYWORD_PROMPT = """The YouTube search "{keyword}" returned 0 relevant results for B-roll footage.
{topic_context}

Suggest ONE alternative search keyword that:
1. Keeps the core concept
2. Uses more common/searchable terms
3. Is likely to find documentary, travel, or stock footage

Respond with ONLY the new search keyword, nothing else."""

    # Prompt for multiple keyword alternatives (post-download remix)
    MULTI_KEYWORD_PROMPT = """You are helping find stock footage on YouTube.
The following search keyword returned ZERO results: "{keyword}"

Generate 3-5 alternative search keywords that are MORE LIKELY to find relevant video footage.

Guidelines:
1. Remove overly specific terms (dates, exact names, niche terminology)
2. Use broader, more visual search terms
3. Focus on what would actually appear in video footage
4. Keep the core concept but make it more searchable
5. Consider YouTube's content library (news clips, documentaries, b-roll)

Original keyword: "{keyword}"
Topic context: {topic_context}

Respond ONLY with a JSON array of alternative keywords, nothing else:
["keyword1", "keyword2", "keyword3"]"""

    def __init__(self, config: 'Config'):
        """Initialize generator.

        Args:
            config: Config object with LLM settings
        """
        self.config = config

    def generate_single_alternative(
        self,
        keyword: str,
        topic: str = "",
        provider: str = "gemini",
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> Optional[str]:
        """Generate a single alternative keyword (for download search optimization).

        Args:
            keyword: Original keyword that returned 0 results
            topic: Optional topic context
            provider: LLM provider ('gemini' or 'anthropic')
            model: Optional model override
            api_key: Optional API key override

        Returns:
            Single alternative keyword, or None if generation failed
        """
        from src.llm_client import create_client, LLMRequest, ResponseFormat

        # Get API key and model from config if not provided
        if not api_key:
            api_key = self._get_api_key(provider)
            if not api_key:
                logger.debug(f"No API key for {provider}, skipping LLM remix")
                return None

        if not model:
            model = self._get_model(provider)

        # Build prompt
        topic_context = f'Topic context: {topic}' if topic else ''
        prompt = self.SINGLE_KEYWORD_PROMPT.format(
            keyword=keyword,
            topic_context=topic_context
        )

        try:
            client = create_client(provider, api_key=api_key, model=model)
            request = LLMRequest(
                prompt=prompt,
                response_format=ResponseFormat.TEXT,
                cache_key_prefix="keyword_alternative_single"
            )
            response = client.generate(request)

            if response.text:
                remix = response.text.strip().strip('"\'')
                if remix and len(remix) < 100:
                    logger.debug(f"    LLM remix: '{keyword}' → '{remix}'")
                    return remix

        except Exception as e:
            logger.debug(f"LLM remix error ({provider}): {e}")

        return None

    def generate_multiple_alternatives(
        self,
        keyword: str,
        topic_context: str = "",
        provider: str = "gemini",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 500
    ) -> Tuple[List[str], str]:
        """Generate multiple alternative keywords (for post-download remix).

        Args:
            keyword: Original keyword that had 0 results
            topic_context: Optional topic context
            provider: LLM provider ('gemini' or 'anthropic')
            model: Optional model override
            api_key: Optional API key override
            max_tokens: Max tokens for response (Anthropic only)

        Returns:
            Tuple of (list of keywords, status message)
        """
        from src.llm_client import create_client, LLMRequest, ResponseFormat

        # Get API key and model from config if not provided
        if not api_key:
            api_key = self._get_api_key(provider)
            if not api_key:
                return [], f"{provider.capitalize()} API key not available"

        if not model:
            model = self._get_model(provider)

        # Build prompt
        prompt = self.MULTI_KEYWORD_PROMPT.format(
            keyword=keyword,
            topic_context=topic_context
        )

        try:
            client = create_client(provider, api_key=api_key, model=model)

            request_kwargs = {
                'prompt': prompt,
                'response_format': ResponseFormat.JSON_ARRAY,
                'cache_key_prefix': "keyword_alternative_multi"
            }

            # Add max_tokens only for Anthropic
            if provider == 'anthropic':
                request_kwargs['max_tokens'] = max_tokens

            request = LLMRequest(**request_kwargs)
            response = client.generate(request)

            if response.parsed_data and isinstance(response.parsed_data, list) and len(response.parsed_data) > 0:
                logger.debug(f"Remixed '{keyword}' → {response.parsed_data}")
                return response.parsed_data, f"{provider.capitalize()} remix successful"

        except Exception as e:
            logger.warning(f"{provider.capitalize()} remix failed for '{keyword}': {e}")

        return [], f"{provider.capitalize()} remix failed"

    def simple_remix_keyword(self, keyword: str) -> Optional[str]:
        """Simple keyword remix without LLM (for download search optimization).

        Strategies:
        1. Add "footage" suffix
        2. Remove qualifiers like "best", "top"
        3. Simplify to core words

        Args:
            keyword: Original keyword

        Returns:
            Remixed keyword or None
        """
        words = keyword.lower().split()

        # Remove common non-searchable qualifiers
        skip_words = {'the', 'a', 'an', 'best', 'top', 'famous', 'popular', 'amazing', 'incredible'}
        core_words = [w for w in words if w not in skip_words]

        if not core_words:
            return None

        # Strategy 1: Add "footage" if not present
        if 'footage' not in keyword.lower() and 'video' not in keyword.lower():
            return ' '.join(core_words[:3]) + ' footage'

        # Strategy 2: Simplify to first 2 core words
        if len(core_words) >= 2:
            return ' '.join(core_words[:2])

        return None

    def fallback_remix(self, keyword: str) -> List[str]:
        """Rule-based fallback when LLM is unavailable (for post-download remix).

        Strategies:
        1. Remove year patterns
        2. Remove "footage" suffix
        3. Remove specific locations
        4. Take word subsets

        Args:
            keyword: Original keyword

        Returns:
            List of up to 3 alternative keywords
        """
        remixed = []

        # Remove year patterns
        no_year = re.sub(r'\b(19|20)\d{2}\b', '', keyword).strip()
        if no_year and no_year != keyword:
            remixed.append(no_year)

        # Remove "footage" suffix
        no_footage = re.sub(r'\s*footage\s*$', '', keyword, flags=re.IGNORECASE).strip()
        if no_footage and no_footage != keyword:
            remixed.append(no_footage)
            remixed.append(f"{no_footage} video")

        # Remove specific locations
        states = ['Kansas', 'Texas', 'California', 'Florida', 'New York', 'Ohio']
        simplified = keyword
        for state in states:
            simplified = re.sub(rf'\b{state}\b', '', simplified, flags=re.IGNORECASE)
        simplified = ' '.join(simplified.split())
        if simplified and simplified != keyword:
            remixed.append(simplified)

        # Take subsets of words
        words = keyword.split()
        if len(words) >= 3:
            remixed.append(' '.join(words[:2]))
            remixed.append(' '.join(words[-2:]))

        return list(dict.fromkeys([r.strip() for r in remixed if r.strip()]))[:3]

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for provider from config.

        Args:
            provider: Provider name ('gemini' or 'anthropic')

        Returns:
            API key or None
        """
        if not self.config:
            return None

        if provider == 'gemini':
            if hasattr(self.config, 'api_keys'):
                return getattr(self.config.api_keys, 'gemini_api_key', None)
        elif provider == 'anthropic':
            if hasattr(self.config, 'api_keys'):
                return getattr(self.config.api_keys, 'anthropic_api_key', None)

        return None

    def _get_model(self, provider: str) -> str:
        """Get model name for provider from config.

        Args:
            provider: Provider name ('gemini' or 'anthropic')

        Returns:
            Model name
        """
        default_models = {
            'gemini': 'gemini-2.0-flash',
            'anthropic': 'claude-3-haiku-20240307'
        }

        if not self.config:
            return default_models.get(provider, 'gemini-2.0-flash')

        # Try matching config first
        if hasattr(self.config, 'matching'):
            if provider == 'gemini':
                return getattr(self.config.matching, 'gemini_model', default_models['gemini'])
            elif provider == 'anthropic':
                return getattr(self.config.matching, 'anthropic_model', default_models['anthropic'])

        # Try llm config
        if hasattr(self.config, 'llm'):
            if provider == 'gemini':
                return getattr(self.config.llm, 'model', default_models['gemini'])
            elif provider == 'anthropic':
                return getattr(self.config.llm, 'anthropic_model', default_models['anthropic'])

        return default_models.get(provider, 'gemini-2.0-flash')
