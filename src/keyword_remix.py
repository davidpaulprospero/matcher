"""
Keyword Remix and Confidence Enforcement Module

Features:
1. 90% confidence requirement - remix keywords for low-confidence matches
2. Zero-download keyword remix - regenerate keywords that return no results
3. Topic-aware keyword generation with context
4. Maximum 3 retries per keyword
"""

import logging
import json
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class KeywordRemixResult:
    """Result of a keyword remix attempt"""
    original_keyword: str
    remixed_keywords: List[str]
    reason: str  # "low_confidence", "zero_downloads"
    attempt: int
    success: bool = False


@dataclass
class ConfidenceEnforcementConfig:
    """Configuration for confidence enforcement"""
    min_confidence: float = 0.90  # 90% minimum confidence
    max_retries: int = 3  # Maximum remix attempts per keyword
    remix_low_confidence: bool = True
    remix_zero_downloads: bool = True
    topic_context: str = ""  # Main topic for context-aware remixing


class KeywordRemixer:
    """
    Remix keywords using Gemini for better results.
    
    Handles:
    - Low-confidence matches -> remix keyword for better footage
    - Zero-download keywords -> generate related alternatives
    - Topic-aware generation (not just dates)
    """
    
    def __init__(self, gemini_api_key: str = None, topic_context: str = ""):
        import os
        self.api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY required for keyword remixing")
        
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        self.topic_context = topic_context
        self.remix_history: Dict[str, List[KeywordRemixResult]] = {}
        
    def set_topic_context(self, context: str):
        """Set the main topic context for all remixes"""
        self.topic_context = context
        logger.info(f"Topic context set: {context[:100]}...")
    
    def remix_for_low_confidence(
        self,
        keyword: str,
        matched_text: str,
        voiceover_text: str,
        confidence: float,
        attempt: int = 1
    ) -> List[str]:
        """
        Remix a keyword that produced low-confidence matches.
        
        Args:
            keyword: Original keyword that was searched
            matched_text: The text of the matched clip
            voiceover_text: The voiceover segment being matched
            confidence: The confidence score achieved
            attempt: Current attempt number (1-3)
        
        Returns:
            List of 2-3 alternative keywords to try
        """
        prompt = f"""A video search keyword produced low-quality matches for documentary footage.

TOPIC CONTEXT: {self.topic_context}

ORIGINAL KEYWORD: "{keyword}"
VOICEOVER TEXT: "{voiceover_text[:200]}"
BEST MATCH FOUND: "{matched_text[:150]}"
MATCH CONFIDENCE: {confidence:.0%} (need 90%+)

The keyword "{keyword}" isn't finding footage that matches the voiceover well.

Generate 2-3 ALTERNATIVE search keywords that would find BETTER matching footage.

Requirements:
- Keywords must relate to the TOPIC CONTEXT, not just generic terms
- If the keyword contains a date/year, KEEP the date but ADD relevant topic terms
  BAD: "2018" -> GOOD: "2018 volcanic eruption footage"
  BAD: "January 2023" -> GOOD: "January 2023 Hawaii lava"
- Focus on VISUAL content that would appear in documentaries
- Be specific but not too narrow
- Consider synonyms and related visual concepts

Respond with ONLY a JSON array of 2-3 keywords, nothing else:
["keyword one", "keyword two", "keyword three"]"""

        try:
            response = self.model.generate_content(prompt)
            
            # Parse response
            text = response.text.strip()
            # Extract JSON array
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                keywords = json.loads(match.group())
                if isinstance(keywords, list) and len(keywords) > 0:
                    # Filter out empty or too short keywords
                    keywords = [k.strip() for k in keywords if len(k.strip()) > 2]
                    
                    # Log the remix
                    result = KeywordRemixResult(
                        original_keyword=keyword,
                        remixed_keywords=keywords,
                        reason="low_confidence",
                        attempt=attempt,
                        success=True
                    )
                    self._track_remix(keyword, result)
                    
                    logger.info(f"Remixed '{keyword}' -> {keywords} (confidence was {confidence:.0%})")
                    return keywords[:3]
        except Exception as e:
            logger.warning(f"Keyword remix failed: {e}")
        
        return []
    
    def remix_for_zero_downloads(
        self,
        keyword: str,
        attempt: int = 1
    ) -> List[str]:
        """
        Remix a keyword that returned zero downloads.
        
        Args:
            keyword: Original keyword with no results
            attempt: Current attempt number (1-3)
        
        Returns:
            List of 2-3 alternative keywords to try
        """
        prompt = f"""A video search keyword returned ZERO results for documentary footage.

TOPIC CONTEXT: {self.topic_context}

FAILED KEYWORD: "{keyword}"
ATTEMPT: {attempt} of 3

The keyword "{keyword}" found NO footage. Generate ALTERNATIVE search keywords.

Requirements:
- Keywords MUST relate to the main topic: {self.topic_context}
- DO NOT generate generic date-only keywords like "2018" or "January"
- If the original has a date, COMBINE it with topic terms:
  BAD: "2018" -> GOOD: "2018 {self.topic_context.split()[0] if self.topic_context else 'event'} footage"
- Use BROADER or MORE COMMON terms that stock footage sites would have
- Think about what a videographer would tag their footage as
- Consider: locations, events, natural phenomena, people activities

Respond with ONLY a JSON array of 2-3 keywords:
["keyword one", "keyword two", "keyword three"]"""

        try:
            response = self.model.generate_content(prompt)
            
            text = response.text.strip()
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                keywords = json.loads(match.group())
                if isinstance(keywords, list) and len(keywords) > 0:
                    keywords = [k.strip() for k in keywords if len(k.strip()) > 2]
                    
                    result = KeywordRemixResult(
                        original_keyword=keyword,
                        remixed_keywords=keywords,
                        reason="zero_downloads",
                        attempt=attempt,
                        success=True
                    )
                    self._track_remix(keyword, result)
                    
                    logger.info(f"Remixed zero-result '{keyword}' -> {keywords}")
                    return keywords[:3]
        except Exception as e:
            logger.warning(f"Zero-download remix failed: {e}")
        
        return []
    
    def remix_batch_keywords(
        self,
        failed_keywords: List[str],
        reason: str = "zero_downloads"
    ) -> Dict[str, List[str]]:
        """
        Remix multiple failed keywords in one batch.
        
        Args:
            failed_keywords: List of keywords that failed
            reason: Why they failed ("zero_downloads" or "low_confidence")
        
        Returns:
            Dict mapping original keyword to list of alternatives
        """
        if not failed_keywords:
            return {}
        
        prompt = f"""Multiple video search keywords failed for documentary footage.

TOPIC CONTEXT: {self.topic_context}

FAILED KEYWORDS:
{chr(10).join(f'- "{k}"' for k in failed_keywords[:10])}

REASON: {reason}

Generate 2 ALTERNATIVE keywords for EACH failed keyword.

Requirements:
- Each alternative MUST relate to the topic: {self.topic_context}
- NO date-only keywords - always combine dates with topic terms
- Use terms that stock footage sites would have
- Be specific enough to find relevant footage

Respond with ONLY a JSON object mapping each keyword to alternatives:
{{"original keyword": ["alt1", "alt2"], "another keyword": ["alt1", "alt2"]}}"""

        try:
            response = self.model.generate_content(prompt)
            
            text = response.text.strip()
            # Extract JSON object
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                if isinstance(result, dict):
                    logger.info(f"Batch remixed {len(result)} keywords")
                    return result
        except Exception as e:
            logger.warning(f"Batch remix failed: {e}")
        
        return {}
    
    def _track_remix(self, original: str, result: KeywordRemixResult):
        """Track remix history"""
        if original not in self.remix_history:
            self.remix_history[original] = []
        self.remix_history[original].append(result)
    
    def get_remix_count(self, keyword: str) -> int:
        """Get number of times a keyword has been remixed"""
        return len(self.remix_history.get(keyword, []))
    
    def get_all_tried_keywords(self, original: str) -> Set[str]:
        """Get all keywords tried for an original (including remixes)"""
        tried = {original}
        for result in self.remix_history.get(original, []):
            tried.update(result.remixed_keywords)
        return tried


class ConfidenceEnforcer:
    """
    Enforce minimum confidence requirements and trigger remixes.
    
    Workflow:
    1. After matching, check all matches below 90% confidence
    2. Group by source keyword
    3. Remix keywords with Gemini
    4. Re-download footage for remixed keywords
    5. Re-match affected segments
    6. Repeat up to 3 times
    """
    
    def __init__(
        self,
        config: ConfidenceEnforcementConfig,
        remixer: KeywordRemixer
    ):
        self.config = config
        self.remixer = remixer
        
        # Tracking
        self.low_confidence_segments: List[Dict] = []
        self.retry_counts: Dict[str, int] = {}  # keyword -> retry count
        self.final_results: Dict[int, float] = {}  # segment_idx -> final confidence
    
    def check_matches(
        self,
        matches: List[Dict],
        keyword_to_segments: Dict[str, List[int]]
    ) -> Tuple[List[str], Dict[str, List[int]]]:
        """
        Check matches and identify keywords needing remix.
        
        Args:
            matches: List of match results with 'confidence' and 'segment_idx'
            keyword_to_segments: Mapping of keywords to segment indices they serve
        
        Returns:
            Tuple of (keywords_to_remix, keyword_to_low_segments)
        """
        # Find low-confidence matches
        low_confidence = []
        for match in matches:
            confidence = match.get('confidence', 0)
            if confidence < self.config.min_confidence:
                low_confidence.append(match)
        
        if not low_confidence:
            logger.info(f"All {len(matches)} matches meet {self.config.min_confidence:.0%} confidence threshold")
            return [], {}
        
        logger.warning(f"{len(low_confidence)}/{len(matches)} matches below {self.config.min_confidence:.0%} confidence")
        
        # Group by keyword
        keyword_to_low_segments: Dict[str, List[int]] = {}
        for match in low_confidence:
            # Find which keyword produced this match's source
            source_keyword = match.get('source_keyword', 'unknown')
            if source_keyword not in keyword_to_low_segments:
                keyword_to_low_segments[source_keyword] = []
            keyword_to_low_segments[source_keyword].append(match.get('segment_idx', -1))
        
        # Filter to keywords that haven't exceeded retry limit
        keywords_to_remix = []
        for keyword in keyword_to_low_segments:
            current_retries = self.retry_counts.get(keyword, 0)
            if current_retries < self.config.max_retries:
                keywords_to_remix.append(keyword)
            else:
                logger.warning(f"Keyword '{keyword}' reached max retries ({self.config.max_retries})")
        
        return keywords_to_remix, keyword_to_low_segments
    
    def check_zero_downloads(
        self,
        keyword_counts: Dict[str, int]
    ) -> List[str]:
        """
        Find keywords with zero downloads that need remixing.
        
        Args:
            keyword_counts: Dict of keyword -> download count
        
        Returns:
            List of keywords to remix
        """
        zero_keywords = [k for k, count in keyword_counts.items() if count == 0]
        
        if not zero_keywords:
            return []
        
        # Filter by retry count
        to_remix = []
        for keyword in zero_keywords:
            current_retries = self.retry_counts.get(keyword, 0)
            if current_retries < self.config.max_retries:
                to_remix.append(keyword)
                self.retry_counts[keyword] = current_retries + 1
            else:
                logger.warning(f"Zero-result keyword '{keyword}' reached max retries")
        
        logger.info(f"Found {len(to_remix)} zero-download keywords to remix")
        return to_remix
    
    def generate_remix_keywords(
        self,
        keywords_to_remix: List[str],
        reason: str = "zero_downloads"
    ) -> Dict[str, List[str]]:
        """
        Generate replacement keywords for failed ones.
        
        Returns:
            Dict mapping original -> list of alternatives
        """
        if not keywords_to_remix:
            return {}
        
        return self.remixer.remix_batch_keywords(keywords_to_remix, reason)


def create_confidence_enforcer(
    min_confidence: float = 0.90,
    max_retries: int = 3,
    topic_context: str = "",
    gemini_api_key: str = None
) -> Tuple[ConfidenceEnforcer, KeywordRemixer]:
    """
    Factory function to create confidence enforcer and remixer.
    
    Args:
        min_confidence: Minimum confidence threshold (default 0.90)
        max_retries: Maximum retry attempts (default 3)
        topic_context: Main topic for context-aware keyword generation
        gemini_api_key: Gemini API key (uses env var if not provided)
    
    Returns:
        Tuple of (ConfidenceEnforcer, KeywordRemixer)
    """
    config = ConfidenceEnforcementConfig(
        min_confidence=min_confidence,
        max_retries=max_retries,
        topic_context=topic_context
    )
    
    remixer = KeywordRemixer(
        gemini_api_key=gemini_api_key,
        topic_context=topic_context
    )
    
    enforcer = ConfidenceEnforcer(config, remixer)
    
    return enforcer, remixer
