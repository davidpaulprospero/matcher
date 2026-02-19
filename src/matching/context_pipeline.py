"""Unified context pipeline with intelligent fallback chain.

This module provides a ContextPipeline class that orchestrates context enrichment
from multiple sources with graceful fallback when higher-quality signals are unavailable.

Fallback chain:
- full_context: All signals available (title, description, tags, chapters, transcript, visual)
- partial_context: Most signals available (title, description, tags, chapters)
- minimal_context: Basic signals available (title, description)
- no_context: No signals available (fallback to segment text only)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from ..utils import SRTSegment

logger = logging.getLogger(__name__)


class ContextLevel(Enum):
    """Context enrichment level based on available signals."""
    FULL = "full"           # All signals available
    PARTIAL = "partial"     # Most signals (title, description, tags, chapters)
    MINIMAL = "minimal"     # Basic signals (title, description)
    NONE = "none"           # No signals available


@dataclass
class ContextEnrichment:
    """Container for enriched context from multiple sources.

    Attributes:
        level: The context level achieved (full, partial, minimal, none)
        context_string: Formatted context string for LLM consumption
        title: Video title (if available)
        description: Video description (if available)
        tags: Video tags (if available)
        chapters: Video chapters (if available)
        transcript_segments: Transcript segments (if available)
        visual_description: Visual description (if available)
        available_signals: List of signal names that were available
        fallback_triggered: Which fallback level triggered (if any)
    """
    level: ContextLevel
    context_string: str
    title: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    chapters: List[Dict[str, Any]] = field(default_factory=list)
    transcript_segments: List[Dict[str, Any]] = field(default_factory=list)
    visual_description: str = ""
    available_signals: List[str] = field(default_factory=list)
    fallback_triggered: Optional[ContextLevel] = None


@dataclass
class ContextPriorityWeightsConfig:
    """Configuration for context priority weights."""
    title: float = 0.35
    description: float = 0.30
    tags: float = 0.20
    chapters: float = 0.15


class ContextPipeline:
    """Unified context pipeline with intelligent fallback chain.

    Provides a single interface for enriching video context from multiple sources,
    gracefully degrading when higher-quality signals are unavailable.

    Usage:
        pipeline = ContextPipeline(video_metadata, transcript_data, config)
        enrichment = pipeline.get_enriched_context(video_id, vo_segment)
    """

    def __init__(
        self,
        video_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        transcript_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        visual_descriptions: Optional[Dict[str, str]] = None,
        priority_weights: Optional[ContextPriorityWeightsConfig] = None,
        use_adaptive_weights: bool = True
    ):
        """Initialize the context pipeline.

        Args:
            video_metadata: Dict mapping video_id to metadata (title, description, tags, chapters)
            transcript_data: Dict mapping video_id to transcript segments
            visual_descriptions: Dict mapping video_id to visual descriptions
            priority_weights: Weights for prioritizing context signals
            use_adaptive_weights: Whether to adapt weights based on available signals
        """
        self.video_metadata = video_metadata or {}
        self.transcript_data = transcript_data or {}
        self.visual_descriptions = visual_descriptions or {}
        self.priority_weights = priority_weights or ContextPriorityWeightsConfig()
        self.use_adaptive_weights = use_adaptive_weights

    def get_enriched_context(
        self,
        video_id: str,
        vo_segment: Optional[SRTSegment] = None
    ) -> ContextEnrichment:
        """Get enriched context for a video with graceful fallback.

        This is the main entry point for context enrichment. It attempts to build
        the richest possible context, falling back through levels when signals
        are unavailable.

        Args:
            video_id: The video ID to get context for
            vo_segment: Optional voiceover segment for additional context

        Returns:
            ContextEnrichment object with available context and metadata
        """
        # Get video metadata
        metadata = self.video_metadata.get(video_id, {})

        # Try full context first
        result = self._build_full_context(video_id, metadata, vo_segment)
        if result.level != ContextLevel.NONE:
            logger.debug(f"ContextPipeline: video_id={video_id}, level=full, signals={result.available_signals}")
            return result

        # Fallback to partial context
        result = self._build_partial_context(video_id, metadata)
        if result.level != ContextLevel.NONE:
            logger.info(f"ContextPipeline: video_id={video_id}, level=partial (fallback), signals={result.available_signals}")
            return result

        # Fallback to minimal context
        result = self._build_minimal_context(video_id, metadata)
        if result.level != ContextLevel.NONE:
            logger.warning(f"ContextPipeline: video_id={video_id}, level=minimal (fallback), signals={result.available_signals}")
            return result

        # Fallback to no context
        result = self._build_no_context(video_id, metadata, vo_segment)
        logger.warning(f"ContextPipeline: video_id={video_id}, level=no_context (fallback)")
        return result

    def _build_full_context(
        self,
        video_id: str,
        metadata: Dict[str, Any],
        vo_segment: Optional[SRTSegment]
    ) -> ContextEnrichment:
        """Build full context: all signals available.

        Requires: title, description, tags OR chapters, AND (transcript OR visual)

        Returns:
            ContextEnrichment with full context or ContextLevel.NONE if insufficient signals
        """
        title = metadata.get('title', '') or ''
        description = metadata.get('description', '') or ''
        tags = metadata.get('tags', []) or []
        chapters = metadata.get('chapters', []) or []
        transcript_segments = self.transcript_data.get(video_id, []) or []
        visual_description = self.visual_descriptions.get(video_id, '') or ''

        # Check which full-context signals are available
        has_basic = bool(title and description)
        has_advanced = bool(tags or chapters)
        has_extra = bool(transcript_segments or visual_description)

        if not has_basic:
            return ContextEnrichment(
                level=ContextLevel.NONE,
                context_string=""
            )

        if not has_advanced and not has_extra:
            return ContextEnrichment(
                level=ContextLevel.NONE,
                context_string=""
            )

        # Build full context
        available_signals = []
        if title:
            available_signals.append('title')
        if description:
            available_signals.append('description')
        if tags:
            available_signals.append('tags')
        if chapters:
            available_signals.append('chapters')
        if transcript_segments:
            available_signals.append('transcript')
        if visual_description:
            available_signals.append('visual')

        # Use adaptive weights if enabled
        priority_weights = self.priority_weights
        if self.use_adaptive_weights:
            priority_weights = self._compute_adaptive_weights(
                title=bool(title),
                description=bool(description),
                tags=bool(tags),
                chapters=bool(chapters),
                transcript=bool(transcript_segments),
                visual=bool(visual_description)
            )

        context_string = self._format_context_string(
            title=title,
            description=description,
            tags=tags,
            chapters=chapters,
            transcript_segments=transcript_segments,
            visual_description=visual_description,
            priority_weights=priority_weights
        )

        return ContextEnrichment(
            level=ContextLevel.FULL,
            context_string=context_string,
            title=title,
            description=description,
            tags=tags if isinstance(tags, list) else [],
            chapters=chapters if isinstance(chapters, list) else [],
            transcript_segments=transcript_segments if isinstance(transcript_segments, list) else [],
            visual_description=visual_description,
            available_signals=available_signals,
            fallback_triggered=None
        )

    def _build_partial_context(
        self,
        video_id: str,
        metadata: Dict[str, Any]
    ) -> ContextEnrichment:
        """Build partial context: title, description, tags, chapters (no transcript/visual).

        Returns:
            ContextEnrichment with partial context or ContextLevel.NONE if insufficient signals
        """
        title = metadata.get('title', '') or ''
        description = metadata.get('description', '') or ''
        tags = metadata.get('tags', []) or []
        chapters = metadata.get('chapters', []) or []

        if not (title and description):
            return ContextEnrichment(
                level=ContextLevel.NONE,
                context_string=""
            )

        # Build partial context
        available_signals = []
        if title:
            available_signals.append('title')
        if description:
            available_signals.append('description')
        if tags:
            available_signals.append('tags')
        if chapters:
            available_signals.append('chapters')

        priority_weights = self.priority_weights
        if self.use_adaptive_weights:
            priority_weights = self._compute_adaptive_weights(
                title=bool(title),
                description=bool(description),
                tags=bool(tags),
                chapters=bool(chapters),
                transcript=False,
                visual=False
            )

        context_string = self._format_context_string(
            title=title,
            description=description,
            tags=tags,
            chapters=chapters,
            transcript_segments=[],
            visual_description="",
            priority_weights=priority_weights
        )

        return ContextEnrichment(
            level=ContextLevel.PARTIAL,
            context_string=context_string,
            title=title,
            description=description,
            tags=tags if isinstance(tags, list) else [],
            chapters=chapters if isinstance(chapters, list) else [],
            available_signals=available_signals,
            fallback_triggered=ContextLevel.FULL
        )

    def _build_minimal_context(
        self,
        video_id: str,
        metadata: Dict[str, Any]
    ) -> ContextEnrichment:
        """Build minimal context: title and description only.

        Returns:
            ContextEnrichment with minimal context or ContextLevel.NONE if insufficient signals
        """
        title = metadata.get('title', '') or ''
        description = metadata.get('description', '') or ''

        if not (title and description):
            return ContextEnrichment(
                level=ContextLevel.NONE,
                context_string=""
            )

        available_signals = ['title', 'description']

        # Minimal context uses default weights
        context_string = f"Video context: Title: {title} | Description: {description[:100]}"

        return ContextEnrichment(
            level=ContextLevel.MINIMAL,
            context_string=context_string,
            title=title,
            description=description,
            available_signals=available_signals,
            fallback_triggered=ContextLevel.PARTIAL
        )

    def _build_no_context(
        self,
        video_id: str,
        metadata: Dict[str, Any],
        vo_segment: Optional[SRTSegment]
    ) -> ContextEnrichment:
        """Build fallback context when no video metadata available.

        Returns:
            ContextEnrichment with no context (empty string)
        """
        return ContextEnrichment(
            level=ContextLevel.NONE,
            context_string="",
            available_signals=[],
            fallback_triggered=ContextLevel.MINIMAL
        )

    def _compute_adaptive_weights(
        self,
        title: bool,
        description: bool,
        tags: bool,
        chapters: bool,
        transcript: bool,
        visual: bool
    ) -> ContextPriorityWeightsConfig:
        """Compute adaptive weights based on available signals.

        When fewer signals are available, the available ones get higher weights.
        """
        # Count available signals
        available_count = sum([
            title, description, tags, chapters, transcript, visual
        ])

        if available_count == 0:
            return ContextPriorityWeightsConfig()

        # Base weights to distribute
        base_weights = {
            'title': 0.35,
            'description': 0.30,
            'tags': 0.20,
            'chapters': 0.15,
            'transcript': 0.10,
            'visual': 0.10
        }

        # Boost available signals
        boost_factor = 6.0 / available_count  # Normalize to full set

        weights = ContextPriorityWeightsConfig()
        if title:
            weights.title = min(1.0, base_weights['title'] * boost_factor)
        if description:
            weights.description = min(1.0, base_weights['description'] * boost_factor)
        if tags:
            weights.tags = min(1.0, base_weights['tags'] * boost_factor)
        if chapters:
            weights.chapters = min(1.0, base_weights['chapters'] * boost_factor)

        return weights

    def _format_context_string(
        self,
        title: str,
        description: str,
        tags: List[str],
        chapters: List[Dict[str, Any]],
        transcript_segments: List[Dict[str, Any]],
        visual_description: str,
        priority_weights: ContextPriorityWeightsConfig
    ) -> str:
        """Format context string from available signals.

        Args:
            title: Video title
            description: Video description
            tags: Video tags
            chapters: Video chapters
            transcript_segments: Transcript segments
            visual_description: Visual description
            priority_weights: Weights for formatting emphasis

        Returns:
            Formatted context string for LLM consumption
        """
        parts = []

        # Weights indicator
        weights_str = (
            f"[weights: title={priority_weights.title:.0%}, "
            f"desc={priority_weights.description:.0%}, "
            f"tags={priority_weights.tags:.0%}, "
            f"chapters={priority_weights.chapters:.0%}]"
        )
        parts.append(f"Video context {weights_str}:")

        # Title (highest priority)
        if title:
            parts.append(f"Title: {title}")

        # Description
        if description:
            # Truncate to first meaningful sentence
            first_sentence = description.split('.')[0].strip()
            if len(first_sentence) > 100:
                first_sentence = first_sentence[:97] + "..."
            if first_sentence:
                parts.append(f"Description: {first_sentence}")

        # Tags
        if tags:
            # Limit to top 5 tags
            relevant_tags = tags[:5] if len(tags) > 5 else tags
            parts.append(f"Tags: {', '.join(relevant_tags)}")

        # Chapters
        if chapters:
            chapter_titles = []
            for ch in chapters[:3]:
                if isinstance(ch, dict):
                    ch_title = ch.get('title', '')
                else:
                    ch_title = str(ch)
                if ch_title and ch_title != 'Unknown':
                    chapter_titles.append(ch_title)
            if chapter_titles:
                parts.append(f"Chapters: {', '.join(chapter_titles)}")

        # Transcript (if available)
        if transcript_segments:
            # Get first few transcript segments as context
            seg_texts = []
            for seg in transcript_segments[:3]:
                if isinstance(seg, dict):
                    text = seg.get('text', '')
                    if text:
                        seg_texts.append(text[:50])
            if seg_texts:
                parts.append(f"Transcript: {' | '.join(seg_texts)}")

        # Visual description (if available)
        if visual_description:
            parts.append(f"Visual: {visual_description[:100]}")

        return " | ".join(parts)

    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about context pipeline usage.

        Returns:
            Dict with stats about video metadata and transcript coverage
        """
        total_videos = len(self.video_metadata)
        videos_with_tags = sum(
            1 for m in self.video_metadata.values()
            if m.get('tags') and len(m.get('tags', [])) > 0
        )
        videos_with_chapters = sum(
            1 for m in self.video_metadata.values()
            if m.get('chapters') and len(m.get('chapters', [])) > 0
        )
        videos_with_transcripts = len(self.transcript_data)
        videos_with_visual = len(self.visual_descriptions)

        return {
            'total_videos': total_videos,
            'videos_with_tags': videos_with_tags,
            'videos_with_chapters': videos_with_chapters,
            'videos_with_transcripts': videos_with_transcripts,
            'videos_with_visual': videos_with_visual
        }
