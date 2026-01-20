#!/usr/bin/env python3
"""
Google Gemini Client with Google Search Grounding.
Compatible with both google-generativeai and google-genai SDKs.
"""

import os
import time
import random
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 2
MAX_BACKOFF = 60


@dataclass
class ResearchResult:
    """Research output with sources"""
    content: str
    sources: List[Dict[str, str]] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    model: str = ""
    topic: str = ""


class GeminiClient:
    """
    Google Gemini API Client with Google Search grounding.
    Supports both google-generativeai and google-genai SDKs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = None,
        enable_search: bool = True
    ):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.model_name = model or os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')
        self.enable_search = enable_search
        self._use_new_sdk = False

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Add to .env or pass api_key.")

        # Try new SDK first, fall back to old SDK
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
            self.client = genai.Client(api_key=self.api_key)
            self._use_new_sdk = True
            logger.info(f"Gemini client initialized (google-genai): {self.model_name}")
        except ImportError:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.genai = genai
                self.model = genai.GenerativeModel(self.model_name)
                self._use_new_sdk = False
                logger.info(f"Gemini client initialized (google-generativeai): {self.model_name}")
            except ImportError:
                raise ImportError(
                    "Neither google-genai nor google-generativeai installed. "
                    "Run: pip install google-generativeai"
                )

    def _generate_with_retry(self, prompt: str, temperature: float = 0.3) -> Any:
        """Generate content with retry logic."""
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                if self._use_new_sdk:
                    # New SDK (google-genai)
                    if self.enable_search:
                        tool = self.types.Tool(google_search=self.types.GoogleSearch())
                        config = self.types.GenerateContentConfig(
                            tools=[tool],
                            temperature=temperature
                        )
                    else:
                        config = self.types.GenerateContentConfig(temperature=temperature)

                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=config
                    )
                else:
                    # Old SDK (google-generativeai)
                    generation_config = {
                        "temperature": temperature,
                        "max_output_tokens": 8192,
                    }
                    response = self.model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )

                return response

            except Exception as e:
                error_str = str(e).lower()
                last_error = e

                retryable = any(code in error_str for code in [
                    '500', '502', '503', '504', '429', 'timeout',
                    'connection', 'unavailable', 'overloaded', 'resource_exhausted'
                ])

                if retryable and attempt < MAX_RETRIES - 1:
                    wait_time = min(INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF)
                    logger.warning(f"Gemini API error: {e}, retrying in {wait_time:.1f}s")
                    time.sleep(wait_time)
                    continue
                else:
                    raise

        raise RuntimeError(f"Gemini API failed after {MAX_RETRIES} attempts: {last_error}")

    def _extract_grounding(self, response) -> tuple:
        """Extract sources and queries from grounding metadata (new SDK only)."""
        sources = []
        queries = []

        if not self._use_new_sdk:
            return sources, queries

        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                meta = candidate.grounding_metadata

                if hasattr(meta, 'web_search_queries'):
                    queries = list(meta.web_search_queries or [])

                if hasattr(meta, 'grounding_chunks'):
                    for chunk in (meta.grounding_chunks or []):
                        if hasattr(chunk, 'web') and chunk.web:
                            sources.append({
                                'url': chunk.web.uri,
                                'title': chunk.web.title or ''
                            })

        return sources, queries

    def _get_text(self, response) -> str:
        """Extract text from response (handles both SDKs)."""
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            return response.candidates[0].content.parts[0].text
        return str(response)

    def research(
        self,
        query: str,
        depth: str = "comprehensive",
        system_prompt: Optional[str] = None
    ) -> ResearchResult:
        """
        Research a topic using Gemini with Google Search grounding.

        Args:
            query: The research question or topic
            depth: "quick" for fast answers, "comprehensive" for deep research
            system_prompt: Optional custom system prompt

        Returns:
            ResearchResult with content and sources
        """
        logger.info(f"Researching with Gemini: {query[:50]}... (depth: {depth})")

        if system_prompt is None:
            system_prompt = """You are a research assistant with access to Google Search.

Provide comprehensive, accurate, and well-sourced information.
- Use specific numbers and facts, not vague statements
- Include source attribution for major claims
- Distinguish between confirmed facts and estimates
- Provide both technical details and practical implications
- Use clear section headings for organization"""

        if depth == "quick":
            user_prompt = f"""Research this topic and provide a concise summary (500-800 words):

{query}

Focus on:
- Key facts and findings
- Current status
- Most important implications
- 3-5 key sources"""
        else:
            user_prompt = f"""Research this topic comprehensively:

{query}

Provide:
1. Executive summary (2-3 paragraphs)
2. Key findings with source attribution
3. Technical details where relevant
4. Current status and developments
5. Practical implications
6. Important caveats or limitations
7. Key sources and references

Use clear section headings. Be thorough but organized."""

        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        response = self._generate_with_retry(full_prompt, temperature=0.3)

        sources, queries = self._extract_grounding(response)
        content = self._get_text(response)

        return ResearchResult(
            content=content,
            sources=sources,
            search_queries=queries,
            model=self.model_name,
            topic=query
        )

    def fact_check(self, claims: List[str]) -> ResearchResult:
        """Verify claims with Google Search."""
        logger.info(f"Fact-checking {len(claims)} claims with Gemini")

        claims_list = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))

        prompt = f"""Fact-check these claims using authoritative sources:

{claims_list}

For each claim provide:
- VERDICT: Confirmed / Partially True / Unverified / False / Outdated
- EVIDENCE: What sources say
- CONTEXT: Important nuances
- SOURCES: Where found"""

        response = self._generate_with_retry(prompt, temperature=0.2)
        sources, _ = self._extract_grounding(response)
        content = self._get_text(response)

        return ResearchResult(
            content=content,
            sources=sources,
            model=self.model_name,
            topic="Fact Check"
        )

    def compare(self, items: List[str], criteria: Optional[List[str]] = None) -> ResearchResult:
        """Compare multiple items with Google Search grounding."""
        logger.info(f"Comparing with Gemini: {', '.join(items)}")

        items_str = "\n- ".join(items)
        criteria_str = ""
        if criteria:
            criteria_str = f"\n\nCompare using these criteria:\n- " + "\n- ".join(criteria)

        prompt = f"""Compare the following:
- {items_str}
{criteria_str}

Provide:
1. Overview of each item
2. Detailed comparison table
3. Pros and cons of each
4. Recommendations for different use cases
5. Sources for claims"""

        response = self._generate_with_retry(prompt, temperature=0.3)
        sources, _ = self._extract_grounding(response)
        content = self._get_text(response)

        return ResearchResult(
            content=content,
            sources=sources,
            model=self.model_name,
            topic=f"Comparison: {', '.join(items)}"
        )

    def api_docs(self, api_name: str, focus: Optional[str] = None) -> ResearchResult:
        """Research API documentation with Google Search."""
        logger.info(f"Researching API with Gemini: {api_name}")

        focus_str = f" Focus specifically on: {focus}" if focus else ""

        prompt = f"""Research the {api_name} API and provide comprehensive documentation.{focus_str}

Include:
1. **Overview**: What the API does and its primary use cases
2. **Authentication**: How to authenticate
3. **Key Endpoints/Methods**: The most important functions with examples
4. **Code Examples**: Working code snippets (Python preferred)
5. **Common Patterns**: Best practices and common usage patterns
6. **Limitations**: Known limitations or gotchas
7. **Error Handling**: Common errors and how to handle them
8. **Rate Limits**: If applicable
9. **Alternatives**: Similar APIs or approaches
10. **Official Documentation Links**

Be specific with method signatures, parameters, and return types."""

        response = self._generate_with_retry(prompt, temperature=0.3)
        sources, _ = self._extract_grounding(response)
        content = self._get_text(response)

        return ResearchResult(
            content=content,
            sources=sources,
            model=self.model_name,
            topic=f"API: {api_name}"
        )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set")
        exit(1)

    client = GeminiClient()
    result = client.research("PySceneDetect scene detection accuracy", depth="quick")
    print(f"# Research: {result.topic}\n\n{result.content}")
    if result.sources:
        print(f"\n## Sources ({len(result.sources)})")
        for s in result.sources[:5]:
            print(f"- {s.get('title', 'N/A')}: {s.get('url', '')}")
