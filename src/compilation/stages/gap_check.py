"""
Gap Check Stage

Verifies all tracks meet duration requirements.
Uses LLM to broaden topic when content is insufficient.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Dict, Any, List

from ..state import CompilationState, GapReport

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


class GapAnalyzer:
    """Analyzes tracks for duration gaps."""

    def analyze(
        self,
        tracks: List[List],
        target_duration: float
    ) -> GapReport:
        """
        Check if all tracks meet target duration.

        Args:
            tracks: List of track clip lists
            target_duration: Target duration per track (seconds)

        Returns:
            GapReport with analysis results
        """
        # Handle empty tracks list
        if not tracks:
            return GapReport(
                complete=False,
                total_shortfall=target_duration * 4,  # Assume 4 tracks needed
                track_shortfalls=[target_duration] * 4,
                new_keywords=[]
            )

        track_durations = [
            sum(clip.actual_duration for clip in track)
            for track in tracks
        ]

        track_shortfalls = [
            max(0, target_duration - duration)
            for duration in track_durations
        ]

        total_shortfall = sum(track_shortfalls)

        # Not complete if any track is empty or has significant shortfall
        any_empty = any(len(track) == 0 for track in tracks)
        complete = total_shortfall == 0 and not any_empty

        return GapReport(
            complete=complete,
            total_shortfall=total_shortfall if not complete else 0,
            track_shortfalls=track_shortfalls,
            new_keywords=[]  # Filled by TopicBroadener if needed
        )


class TopicBroadener:
    """
    Uses LLM to generate broader/related keywords when content is insufficient.
    """

    def __init__(self, config: 'Config'):
        self.config = config
        self._llm_client = None

    @property
    def llm_client(self):
        """Lazy-load LLM client."""
        if self._llm_client is None:
            from src.llm_client import get_llm_client
            self._llm_client = get_llm_client(self.config)
        return self._llm_client

    def broaden(
        self,
        topic: str,
        used_keywords: List[str],
        shortfall_seconds: float,
        compilation_config: Dict[str, Any]
    ) -> List[str]:
        """
        Generate broader keywords to find more content.

        Args:
            topic: Original topic
            used_keywords: Keywords already tried
            shortfall_seconds: How many more seconds of content needed
            compilation_config: Compilation config dict

        Returns:
            List of new keywords to try
        """
        llm_config = compilation_config.get('llm', {})
        provider = llm_config.get('provider', 'gemini')

        shortfall_minutes = shortfall_seconds / 60

        prompt = f"""You are helping find video content for a compilation video.

Topic: {topic}
Keywords already tried: {', '.join(used_keywords)}
Need approximately {shortfall_minutes:.1f} more minutes of video content.

Generate 5 new search keywords that would find related video content.
The keywords should be:
1. Broader or tangential to the topic (not just synonyms)
2. Likely to have entertaining video content on YouTube
3. Different enough from already-tried keywords to find new videos
4. Good for finding 5-60 second clips

Return ONLY a JSON array of 5 keyword strings, no other text.
Example: ["keyword 1", "keyword 2", "keyword 3", "keyword 4", "keyword 5"]"""

        try:
            response = self._call_llm(prompt, provider)

            # Parse response as JSON
            keywords = self._parse_keywords(response)

            # Filter out keywords already used
            new_keywords = [
                kw for kw in keywords
                if kw.lower() not in [uk.lower() for uk in used_keywords]
            ]

            logger.info(f"  TopicBroadener generated: {new_keywords}")
            return new_keywords[:5]  # Limit to 5

        except Exception as e:
            logger.error(f"TopicBroadener error: {e}")
            # Fallback: generate simple variations
            return self._generate_fallback_keywords(topic, used_keywords)

    def _call_llm(self, prompt: str, provider: str) -> str:
        """Call LLM with prompt."""
        try:
            # Use the unified LLM client
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt="You generate YouTube search keywords. Return only JSON arrays.",
                max_tokens=200
            )
            return response

        except Exception as e:
            logger.warning(f"LLM call failed ({provider}): {e}")
            raise

    def _parse_keywords(self, response: str) -> List[str]:
        """Parse keywords from LLM response."""
        # Clean response
        text = response.strip()

        # Try to find JSON array in response
        import re
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            try:
                keywords = json.loads(match.group())
                if isinstance(keywords, list):
                    return [str(k).strip() for k in keywords if k]
            except json.JSONDecodeError:
                pass

        # Fallback: split by newlines or commas
        keywords = []
        for line in text.split('\n'):
            line = line.strip().strip('-').strip('*').strip('"').strip("'")
            if line and len(line) < 50:
                keywords.append(line)

        return keywords[:5]

    def _generate_fallback_keywords(
        self,
        topic: str,
        used_keywords: List[str]
    ) -> List[str]:
        """Generate simple fallback keywords without LLM."""
        words = topic.lower().split()

        # Add common suffixes/variations
        suffixes = ["compilation", "funny", "best", "moments", "clips"]
        prefixes = ["viral", "epic", "amazing", "top"]

        fallback = []
        for word in words:
            for suffix in suffixes:
                kw = f"{word} {suffix}"
                if kw not in used_keywords:
                    fallback.append(kw)
                    break
            for prefix in prefixes:
                kw = f"{prefix} {word}"
                if kw not in used_keywords:
                    fallback.append(kw)
                    break

        return fallback[:5]


class GapCheckStage:
    """
    Check if tracks meet duration requirements.
    Generate new keywords if content is insufficient.
    """

    name = "GAP_CHECK"
    description = "Check track completeness and broaden topic if needed"

    def __init__(self, config: 'Config'):
        self.config = config
        self.gap_analyzer = GapAnalyzer()
        self.topic_broadener = TopicBroadener(config)

    def run(
        self,
        state: CompilationState,
        compilation_config: Dict[str, Any]
    ) -> GapReport:
        """
        Analyze gaps and generate new keywords if needed.

        Args:
            state: Compilation state
            compilation_config: Compilation config dict

        Returns:
            GapReport with analysis and new keywords if applicable
        """
        logger.info("[GAP_CHECK] Analyzing track completeness")

        # Analyze current tracks
        report = self.gap_analyzer.analyze(
            tracks=state.arranged_tracks,
            target_duration=state.target_duration
        )

        if report.complete:
            logger.info("  All tracks complete!")
            return report

        # Log shortfalls
        logger.warning(f"  Total shortfall: {report.total_shortfall:.0f}s")
        for i, shortfall in enumerate(report.track_shortfalls):
            if shortfall > 0:
                logger.warning(f"    Track {i+1}: needs {shortfall:.0f}s more")

        # Generate broader keywords
        max_retries = compilation_config.get('max_retries', 3)
        if state.retry_count < max_retries:
            logger.info("  Generating broader keywords...")
            new_keywords = self.topic_broadener.broaden(
                topic=state.topic,
                used_keywords=state.all_used_keywords,
                shortfall_seconds=report.total_shortfall,
                compilation_config=compilation_config
            )
            report.new_keywords = new_keywords
        else:
            logger.warning(f"  Max retries ({max_retries}) reached, not generating more keywords")

        return report
