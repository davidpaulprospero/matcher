"""
Download search optimization with adaptive pool sizing and keyword alternatives.

Migrated from VideoDownloader keyword remix methods (lines 1610-1906).
Uses unified src.llm_client (Rule 9 compliant).

NOTE: Renamed from KeywordRemixer to SearchOptimizer (Jan 6, 2026) to avoid
naming collision with src/keyword_remix.py KeywordRemixer (different purpose).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)


class SearchOptimizer:
    """Optimizes download searches with adaptive pool sizing and keyword alternatives.

    Handles:
    - Generating alternative keywords when searches return zero results
    - Adaptive search pool sizing based on historical pass rates
    - Inter-keyword source diversity tracking

    Migrated from VideoDownloader keyword remix methods.
    Uses unified src.llm_client (Rule 9 compliant).
    """

    def __init__(self, config: 'Config', llm_call_gemini_func, llm_call_anthropic_func):
        """
        Initialize SearchOptimizer.

        Args:
            config: Config object with download.zero_download_remix settings
            llm_call_gemini_func: Function to call Gemini (from TitleFilter)
            llm_call_anthropic_func: Function to call Anthropic (from TitleFilter)
        """
        self.config = config
        self.download_config = config.download
        self._call_gemini = llm_call_gemini_func
        self._call_anthropic = llm_call_anthropic_func

        # Adaptive search pool tracking
        self._search_pass_rates: Dict[str, List[float]] = defaultdict(list)

        # Inter-keyword source diversity tracking
        self._keyword_sources: Dict[str, Set[str]] = {}  # keyword -> set of video IDs
        self._source_keywords: Dict[str, List[str]] = {}  # video_id -> list of keywords

    def get_remix_keyword(self, keyword: str, topic: str = "") -> Optional[str]:
        """
        Generate an alternative keyword using LLM when original returns 0 results.

        Migrated from downloader.py lines 1610-1671.
        Delegates to unified KeywordAlternativeGenerator.

        Args:
            keyword: Original keyword that returned 0 results
            topic: Optional topic context

        Returns:
            Remixed keyword, or None if remix not possible/enabled
        """
        from src.keyword_alternatives import KeywordAlternativeGenerator

        # Check if remix is enabled
        remix_config = getattr(self.download_config, 'zero_download_remix', None)
        if not remix_config:
            # Default: try simple remix without LLM
            generator = KeywordAlternativeGenerator(self.config)
            return generator.simple_remix_keyword(keyword)

        if isinstance(remix_config, dict):
            enabled = remix_config.get('enabled', True)
            use_llm = remix_config.get('use_llm', False)
        else:
            enabled = getattr(remix_config, 'enabled', True)
            use_llm = getattr(remix_config, 'use_llm', False)

        if not enabled:
            return None

        generator = KeywordAlternativeGenerator(self.config)

        if not use_llm:
            return generator.simple_remix_keyword(keyword)

        # LLM-based remix
        llm_config = getattr(self.download_config, 'llm_title_filter', None)
        if not llm_config:
            return generator.simple_remix_keyword(keyword)

        provider = getattr(llm_config, 'provider', 'gemini')
        model = getattr(llm_config, 'model', 'gemini-2.0-flash')

        # Try LLM remix
        remix = generator.generate_single_alternative(
            keyword=keyword,
            topic=topic,
            provider=provider,
            model=model
        )

        if remix:
            return remix

        # Fallback to simple remix
        return generator.simple_remix_keyword(keyword)

    def simple_remix_keyword(self, keyword: str) -> Optional[str]:
        """
        Simple keyword remix without LLM.

        Migrated from downloader.py lines 1673-1699.
        Delegates to unified KeywordAlternativeGenerator.

        Args:
            keyword: Original keyword

        Returns:
            Remixed keyword or None
        """
        from src.keyword_alternatives import KeywordAlternativeGenerator

        generator = KeywordAlternativeGenerator(self.config)
        return generator.simple_remix_keyword(keyword)

    def get_keyword_category(self, keyword: str) -> str:
        """
        Categorize keyword for adaptive pool sizing.

        Migrated from downloader.py lines 1701-1721.

        Categories help group similar keywords that likely have similar pass rates.

        Args:
            keyword: Keyword to categorize

        Returns:
            Category name
        """
        kw_lower = keyword.lower()

        # Check for common keyword patterns
        if any(term in kw_lower for term in ['footage', 'stock', 'b-roll', 'broll']):
            return 'stock_footage'
        elif any(term in kw_lower for term in ['documentary', 'history', 'explained']):
            return 'documentary'
        elif any(term in kw_lower for term in ['tour', 'walkthrough', 'walk through', 'travel']):
            return 'travel'
        elif any(term in kw_lower for term in ['aerial', 'drone', '4k', 'timelapse']):
            return 'cinematic'
        elif any(term in kw_lower for term in ['interview', 'speech', 'talk']):
            return 'interview'
        else:
            return 'general'

    def get_adaptive_search_pool(self, keyword: str, max_downloads: int) -> int:
        """
        Calculate adaptive search pool size based on historical pass rates.

        Migrated from downloader.py lines 1723-1769.

        Args:
            keyword: The search keyword
            max_downloads: Target number of downloads

        Returns:
            Optimized search pool size
        """
        # Get config defaults
        multiplier = getattr(self.download_config, 'search_pool_multiplier', 5)
        min_pool = getattr(self.download_config, 'min_search_pool', 30)
        max_pool = getattr(self.download_config, 'max_search_pool', 100)

        # Default pool size
        default_pool = max(max_downloads * multiplier, min_pool)

        # Get category and historical pass rates
        category = self.get_keyword_category(keyword)
        pass_rates = self._search_pass_rates.get(category, [])

        if len(pass_rates) < 2:
            # Not enough data yet, use default
            return default_pool

        # Calculate average pass rate for this category
        avg_pass_rate = sum(pass_rates[-10:]) / len(pass_rates[-10:])  # Use last 10

        if avg_pass_rate <= 0.01:
            # Very low pass rate - expand significantly
            adaptive_pool = min(max_pool, default_pool * 3)
            logger.debug(f"    Adaptive pool: {adaptive_pool} (low pass rate {avg_pass_rate:.1%} for '{category}')")
        elif avg_pass_rate < 0.2:
            # Low pass rate - expand pool
            adaptive_pool = min(max_pool, int(default_pool * 2))
            logger.debug(f"    Adaptive pool: {adaptive_pool} (pass rate {avg_pass_rate:.1%} for '{category}')")
        elif avg_pass_rate > 0.6:
            # High pass rate - can use smaller pool
            adaptive_pool = max(min_pool, int(default_pool * 0.7))
            logger.debug(f"    Adaptive pool: {adaptive_pool} (high pass rate {avg_pass_rate:.1%} for '{category}')")
        else:
            # Normal pass rate - use default
            adaptive_pool = default_pool

        return adaptive_pool

    def record_search_pass_rate(self, keyword: str, searched: int, approved: int):
        """
        Record pass rate for adaptive pool sizing.

        Migrated from downloader.py lines 1771-1793.

        Args:
            keyword: The search keyword
            searched: Number of videos searched
            approved: Number that passed LLM filter
        """
        if searched <= 0:
            return

        category = self.get_keyword_category(keyword)
        pass_rate = approved / searched

        if category not in self._search_pass_rates:
            self._search_pass_rates[category] = []

        self._search_pass_rates[category].append(pass_rate)

        # Keep only last 50 entries per category
        if len(self._search_pass_rates[category]) > 50:
            self._search_pass_rates[category] = self._search_pass_rates[category][-50:]

    def record_source_for_keyword(self, keyword: str, video_id: str):
        """
        Record that a video was downloaded for a keyword.

        Migrated from downloader.py lines 1795-1810.

        Used for inter-keyword source diversity analysis.

        Args:
            keyword: Keyword that triggered download
            video_id: YouTube video ID
        """
        # Track keyword -> sources
        if keyword not in self._keyword_sources:
            self._keyword_sources[keyword] = set()
        self._keyword_sources[keyword].add(video_id)

        # Track source -> keywords
        if video_id not in self._source_keywords:
            self._source_keywords[video_id] = []
        if keyword not in self._source_keywords[video_id]:
            self._source_keywords[video_id].append(keyword)

    def get_source_diversity_report(self) -> Dict[str, Any]:
        """
        Generate inter-keyword source diversity report.

        Migrated from downloader.py lines 1812-1848.

        Returns:
            Dict with diversity metrics and overlap warnings
        """
        report = {
            'total_keywords': len(self._keyword_sources),
            'total_unique_sources': len(self._source_keywords),
            'sources_per_keyword': {},
            'overlap_warnings': [],
            'heavily_reused_sources': [],
        }

        # Calculate sources per keyword
        for kw, sources in self._keyword_sources.items():
            report['sources_per_keyword'][kw] = len(sources)

        # Find overlap: sources used by multiple keywords
        for video_id, keywords in self._source_keywords.items():
            if len(keywords) > 1:
                report['overlap_warnings'].append({
                    'video_id': video_id,
                    'keywords': keywords,
                    'count': len(keywords)
                })

        # Sort overlaps by count
        report['overlap_warnings'].sort(key=lambda x: -x['count'])

        # Find heavily reused sources (used by 3+ keywords)
        report['heavily_reused_sources'] = [
            o for o in report['overlap_warnings'] if o['count'] >= 3
        ]

        return report

    def log_source_diversity_report(self):
        """
        Log the source diversity report after download phase.

        Migrated from downloader.py lines 1850-1880.
        """
        report = self.get_source_diversity_report()

        if report['total_keywords'] == 0:
            return

        logger.info("=" * 60)
        logger.info("INTER-KEYWORD SOURCE DIVERSITY REPORT")
        logger.info("=" * 60)
        logger.info(f"  Total keywords: {report['total_keywords']}")
        logger.info(f"  Total unique video sources: {report['total_unique_sources']}")

        # Average sources per keyword
        if report['sources_per_keyword']:
            avg_sources = sum(report['sources_per_keyword'].values()) / len(report['sources_per_keyword'])
            logger.info(f"  Average sources per keyword: {avg_sources:.1f}")

        # Overlap warnings
        overlap_count = len(report['overlap_warnings'])
        if overlap_count > 0:
            logger.warning(f"  Source overlap detected: {overlap_count} videos used by multiple keywords")
            for o in report['overlap_warnings'][:5]:  # Show top 5
                logger.warning(f"    {o['video_id']}: used by {o['count']} keywords ({', '.join(o['keywords'][:3])}...)")

        # Heavy reuse warnings
        if report['heavily_reused_sources']:
            logger.warning(f"  Heavily reused sources (3+ keywords): {len(report['heavily_reused_sources'])}")
            logger.warning("  Consider diversifying keywords or downloading more videos")

        logger.info("=" * 60)

    def get_retry_keyword(self, keyword: str, retry_count: int) -> str:
        """
        Generate alternative keyword for retry after timeout.

        Migrated from downloader.py lines 1882-1906.

        Args:
            keyword: Original keyword that timed out
            retry_count: Which retry this is (0 = first retry, 1 = second retry)

        Returns:
            Modified keyword, or original if no modification possible
        """
        words = keyword.split()

        if retry_count == 0:
            # First retry: simplify by taking first 3 words + "footage"
            if len(words) > 3:
                return ' '.join(words[:3]) + " footage"
            elif "footage" not in keyword.lower():
                return keyword + " footage"

        elif retry_count == 1:
            # Second retry: just the core concept (first 2 words)
            if len(words) >= 2:
                return ' '.join(words[:2])

        return keyword
