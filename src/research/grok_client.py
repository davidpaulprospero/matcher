#!/usr/bin/env python3
"""
Grok (xAI) Client for Research.
Uses OpenAI-compatible API with live web search via X/Twitter.
"""

import os
import time
import random
import logging
import requests
from typing import Optional, List, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 2
MAX_BACKOFF = 60
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass
class ResearchResult:
    """Research result from Grok"""
    content: str
    sources: List[str] = field(default_factory=list)
    model: str = ""
    topic: str = ""


class GrokClient:
    """
    Grok API Client for research with live X/Twitter search.
    Uses xAI's OpenAI-compatible API endpoint.
    """

    BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "grok-3-latest"
    ):
        self.api_key = api_key or os.getenv('XAI_API_KEY')
        if not self.api_key:
            raise ValueError("XAI_API_KEY not set. Add to .env or pass api_key.")

        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        logger.info(f"Grok client initialized (model: {model})")

    def research(
        self,
        query: str,
        depth: str = "comprehensive",
        enable_search: bool = True,
        system_prompt: Optional[str] = None
    ) -> ResearchResult:
        """
        Research a topic using Grok with live X/Twitter search.

        Args:
            query: The research question or topic
            depth: "quick" for fast answers, "comprehensive" for deep research
            enable_search: Enable live web/X search
            system_prompt: Optional custom system prompt

        Returns:
            ResearchResult with content and sources
        """
        logger.info(f"Researching with Grok: {query[:50]}... (depth: {depth})")

        if system_prompt is None:
            system_prompt = """You are a research assistant with access to real-time X (Twitter) data and web information.

Your role is to provide comprehensive, accurate research with:
- Verified facts from official sources and credible news
- Real-time updates from social media when relevant
- Specific dates, numbers, statistics
- Source attribution for major claims

Be thorough but organized. Use clear section headings."""

        if depth == "quick":
            user_prompt = f"""Research this topic and provide a concise summary (500-800 words):

{query}

Focus on:
- Key facts and current status
- Recent developments (especially from X/Twitter if relevant)
- Most important implications
- 3-5 key sources"""
        else:
            user_prompt = f"""Research this topic comprehensively:

{query}

Provide:
1. Executive summary (2-3 paragraphs)
2. Key findings with source attribution
3. Technical details where relevant
4. Current status and recent developments
5. Real-time social media insights if relevant
6. Practical implications
7. Important caveats or limitations
8. Key sources and references

Use clear section headings. Include specific numbers, dates, and statistics."""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 6000 if depth == "comprehensive" else 2000
        }

        if enable_search:
            payload["search"] = {"enabled": True}

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    f"{self.BASE_URL}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=(120, 600)
                )

                if response.status_code in RETRYABLE_STATUS_CODES:
                    wait_time = min(INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF)
                    logger.warning(f"Grok API returned {response.status_code}, retrying in {wait_time:.1f}s")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()

                content = data["choices"][0]["message"]["content"]
                sources = self._extract_sources(content)

                return ResearchResult(
                    content=content,
                    sources=sources,
                    model=self.model,
                    topic=query
                )

            except requests.exceptions.Timeout as e:
                last_error = e
                wait_time = min(INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF)
                logger.warning(f"Grok API timeout, retrying in {wait_time:.1f}s")
                time.sleep(wait_time)

            except requests.exceptions.ConnectionError as e:
                last_error = e
                wait_time = min(INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF)
                logger.warning(f"Grok connection error, retrying in {wait_time:.1f}s")
                time.sleep(wait_time)

            except requests.exceptions.HTTPError as e:
                error_detail = ""
                try:
                    error_detail = response.json()
                except:
                    error_detail = response.text
                raise Exception(f"Grok API error: {e} - {error_detail}")

        raise Exception(f"Grok API failed after {MAX_RETRIES} attempts: {last_error}")

    def _extract_sources(self, content: str) -> List[str]:
        """Extract source references from content."""
        import re
        sources = []

        patterns = [
            r'according to ([A-Z][^,\.]+)',
            r'reported by ([A-Z][^,\.]+)',
            r'source: ([^\n]+)',
            r'\[([^\]]+)\]',
            r'@(\w+)',  # X/Twitter handles
            r'https?://[^\s\)]+',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            sources.extend(matches)

        seen = set()
        unique = []
        for s in sources:
            s_clean = s.strip()
            if s_clean and s_clean.lower() not in seen:
                seen.add(s_clean.lower())
                unique.append(s_clean)

        return unique[:15]

    def compare(self, items: List[str], criteria: Optional[List[str]] = None) -> ResearchResult:
        """Compare multiple items/technologies/approaches."""
        logger.info(f"Comparing with Grok: {', '.join(items)}")

        items_str = "\n- ".join(items)
        criteria_str = ""
        if criteria:
            criteria_str = f"\n\nCompare using these criteria:\n- " + "\n- ".join(criteria)

        user_prompt = f"""Compare the following:
- {items_str}
{criteria_str}

Provide:
1. Overview of each item
2. Detailed comparison table
3. Pros and cons of each
4. Recommendations for different use cases
5. Sources for claims

Use your real-time search to find the latest information."""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.3,
            "max_tokens": 4000,
            "search": {"enabled": True}
        }

        response = requests.post(
            f"{self.BASE_URL}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=(120, 600)
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        sources = self._extract_sources(content)

        return ResearchResult(
            content=content,
            sources=sources,
            model=self.model,
            topic=f"Comparison: {', '.join(items)}"
        )

    def test_connection(self) -> bool:
        """Test API connectivity."""
        try:
            response = requests.get(
                f"{self.BASE_URL}/models",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Grok API connection failed: {e}")
            return False


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('XAI_API_KEY')
    if not api_key:
        print("XAI_API_KEY not set")
        exit(1)

    client = GrokClient()
    result = client.research("latest developments in AI video generation", depth="quick")
    print(f"# Research: {result.topic}\n\n{result.content}")
