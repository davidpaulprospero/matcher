#!/usr/bin/env python3
"""
Perplexity AI Client for Research.
Uses Sonar models for real-time web research.
"""

import os
import time
import random
import requests
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 5
INITIAL_BACKOFF = 2  # seconds
MAX_BACKOFF = 60  # seconds
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass
class ResearchResult:
    """Container for research results"""
    content: str
    sources: List[str] = field(default_factory=list)
    model: str = ""
    topic: str = ""


class PerplexityClient:
    """
    Perplexity AI API Client.
    Uses Sonar models for real-time web research.
    """

    BASE_URL = "https://api.perplexity.ai/chat/completions"

    # Current models (2024+)
    MODELS = {
        "research": "sonar-pro",       # Best for deep research
        "quick": "sonar",              # Standard model
        "reasoning": "sonar-reasoning" # Complex reasoning
    }

    def __init__(self, api_key: Optional[str] = None, default_model: str = "research"):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not set. Add to .env or pass api_key.")

        self.default_model = self.MODELS.get(default_model, default_model)

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        logger.info(f"Perplexity client initialized with model: {self.default_model}")

    def _make_request(self, messages: List[Dict], model: Optional[str] = None) -> Dict:
        """Make API request to Perplexity with retry logic."""
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.9
        }

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    self.BASE_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=(120, 900)  # (connect, read) - 15 min for long research
                )

                # Check if we should retry
                if response.status_code in RETRYABLE_STATUS_CODES:
                    wait_time = min(INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF)
                    logger.warning(f"Perplexity API returned {response.status_code}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout as e:
                last_error = e
                wait_time = min(INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF)
                logger.warning(f"Perplexity API timeout, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                continue

            except requests.exceptions.ConnectionError as e:
                last_error = e
                wait_time = min(INITIAL_BACKOFF * (2 ** attempt) + random.uniform(0, 1), MAX_BACKOFF)
                logger.warning(f"Perplexity connection error, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait_time)
                continue

            except requests.exceptions.HTTPError as e:
                error_detail = ""
                try:
                    error_detail = response.json()
                except:
                    error_detail = response.text
                logger.error(f"Perplexity API error: {e}")
                logger.error(f"Response: {error_detail}")
                raise Exception(f"Perplexity API error: {e} - {error_detail}")

        # All retries exhausted
        raise Exception(f"Perplexity API failed after {MAX_RETRIES} attempts: {last_error}")

    def research(
        self,
        query: str,
        depth: str = "comprehensive",
        system_prompt: Optional[str] = None
    ) -> ResearchResult:
        """
        Research a topic using Perplexity's Sonar models.

        Args:
            query: The research question or topic
            depth: "quick" for fast answers, "comprehensive" for deep research
            system_prompt: Optional custom system prompt

        Returns:
            ResearchResult with content and sources
        """
        logger.info(f"Researching: {query[:50]}... (depth: {depth})")

        if system_prompt is None:
            system_prompt = """You are a research assistant. Provide comprehensive, accurate, and well-sourced information.

Guidelines:
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

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        model = self.MODELS["research"] if depth == "comprehensive" else self.MODELS["quick"]
        result = self._make_request(messages, model=model)

        content = result["choices"][0]["message"]["content"]
        citations = result.get("citations", [])

        return ResearchResult(
            content=content,
            sources=citations if isinstance(citations, list) else [],
            model=model,
            topic=query
        )

    def fact_check(self, claims: List[str]) -> ResearchResult:
        """Verify specific claims."""
        logger.info(f"Fact-checking {len(claims)} claims")

        claims_list = "\n".join([f"{i+1}. {claim}" for i, claim in enumerate(claims)])
        user_prompt = f"""Verify the following claims:

{claims_list}

For each claim, provide:
1. **Verdict**: Confirmed / Partially True / Unverified / False / Outdated
2. **Current Information**: The accurate, up-to-date fact
3. **Source**: Where this can be verified
4. **Context**: Important nuances"""

        messages = [{"role": "user", "content": user_prompt}]

        result = self._make_request(messages)
        citations = result.get("citations", [])

        return ResearchResult(
            content=result["choices"][0]["message"]["content"],
            sources=citations if isinstance(citations, list) else [],
            model=self.default_model,
            topic="Fact Check"
        )

    def compare(self, items: List[str], criteria: Optional[List[str]] = None) -> ResearchResult:
        """Compare multiple items/technologies/approaches."""
        logger.info(f"Comparing: {', '.join(items)}")

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
5. Sources for claims"""

        messages = [{"role": "user", "content": user_prompt}]

        result = self._make_request(messages, model=self.MODELS["research"])
        citations = result.get("citations", [])

        return ResearchResult(
            content=result["choices"][0]["message"]["content"],
            sources=citations if isinstance(citations, list) else [],
            model=self.MODELS["research"],
            topic=f"Comparison: {', '.join(items)}"
        )

    def api_docs(self, api_name: str, focus: Optional[str] = None) -> ResearchResult:
        """Research API documentation and usage patterns."""
        logger.info(f"Researching API: {api_name}")

        focus_str = f" Focus specifically on: {focus}" if focus else ""

        user_prompt = f"""Research the {api_name} API and provide comprehensive documentation.{focus_str}

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

        messages = [{"role": "user", "content": user_prompt}]

        result = self._make_request(messages, model=self.MODELS["research"])
        citations = result.get("citations", [])

        return ResearchResult(
            content=result["choices"][0]["message"]["content"],
            sources=citations if isinstance(citations, list) else [],
            model=self.MODELS["research"],
            topic=f"API: {api_name}"
        )


def research_cli():
    """CLI entry point for research."""
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Research using Perplexity AI")
    parser.add_argument("query", help="Research query or topic")
    parser.add_argument("--depth", choices=["quick", "comprehensive"], default="comprehensive",
                       help="Research depth (default: comprehensive)")
    parser.add_argument("--api", action="store_true", help="Research as API documentation")
    parser.add_argument("--compare", action="store_true", help="Compare items (comma-separated query)")
    parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    try:
        client = PerplexityClient()

        if args.api:
            result = client.api_docs(args.query)
        elif args.compare:
            items = [item.strip() for item in args.query.split(",")]
            result = client.compare(items)
        else:
            result = client.research(args.query, depth=args.depth)

        output = f"# Research: {result.topic}\n\n{result.content}"
        if result.sources:
            output += "\n\n## Sources\n" + "\n".join([f"- {s}" for s in result.sources])

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Saved to: {args.output}")
        else:
            print(output)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(research_cli())
