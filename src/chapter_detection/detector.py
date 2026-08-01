"""
Enhanced Chapter Detector

Main orchestrator for multi-pass chapter detection.
"""

import logging
from typing import List, Dict, Any, Optional

from .models import ChapterCandidate, DetectionResult
from .passes.initial import run_initial_detection, detect_content_type
from .passes.refinement import run_boundary_refinement, compute_voiceover_embeddings
from .passes.validation import run_validation
from .passes.coverage import run_coverage_resolution
from .prompts import format_summary_title_prompt

logger = logging.getLogger(__name__)


class EnhancedChapterDetector:
    """
    Multi-pass chapter detection with confidence scoring.

    Orchestrates four detection passes:
    1. Initial Detection - Detect chapters using topic/location strategy
    2. Boundary Refinement - Adjust boundaries using embedding similarity
    3. Cross-Validation - Verify chapters with independent LLM call
    4. Coverage Resolution - Ensure complete, non-overlapping coverage
    """

    def __init__(self, config: Any):
        """
        Initialize the detector.

        Args:
            config: Configuration object with chapter_detection settings
        """
        self.config = config
        self.llm_client = None
        self._init_llm_client()

    def _init_llm_client(self) -> None:
        """Initialize LLM client from config."""
        try:
            from src.llm_client import create_client

            # Try Gemini first
            api_key = getattr(self.config, 'gemini_api_key', None)
            if not api_key:
                import os
                api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')

            if api_key:
                model = getattr(self.config, 'gemini_model', 'gemini-2.5-flash')
                self.llm_client = create_client("gemini", api_key=api_key, model=model)
                logger.debug("Initialized Gemini LLM client for chapter detection")
                return

            # Try Anthropic
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if api_key:
                self.llm_client = create_client("anthropic", api_key=api_key)
                logger.debug("Initialized Anthropic LLM client for chapter detection")
                return

            logger.warning("No LLM API key available for chapter detection")

        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}")

    def detect_chapters(
        self,
        segments: List[Dict[str, Any]],
        content_type: str = 'auto',
        location_service: Any = None,
        overall_topic: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect chapters in voiceover segments.

        Args:
            segments: List of segment dicts with 'index', 'text', 'start', 'end'
            content_type: Content type ('auto', 'topic', 'location', 'travel', etc.)
            location_service: Optional LocationService for geocoding
            overall_topic: Optional topic context

        Returns:
            List of chapter dicts (backward compatible with LocationChapter)
        """
        result = self.detect_chapters_full(
            segments=segments,
            content_type=content_type,
            location_service=location_service,
            overall_topic=overall_topic,
        )

        # Convert to dicts for backward compatibility
        return [ch.to_dict() for ch in result.chapters]

    def detect_chapters_full(
        self,
        segments: List[Dict[str, Any]],
        content_type: str = 'auto',
        location_service: Any = None,
        overall_topic: str = None,
    ) -> DetectionResult:
        """
        Detect chapters with full result details.

        Args:
            segments: List of segment dicts
            content_type: Content type ('auto', 'topic', 'location', etc.)
            location_service: Optional LocationService for geocoding
            overall_topic: Optional topic context

        Returns:
            DetectionResult with chapters and metadata
        """
        if not segments:
            logger.info("Chapter detection started with empty segments list")
            return DetectionResult(
                chapters=[],
                total_segments=0,
                detection_passes_run=[],
            )

        total_segments = len(segments)
        logger.info(f"Chapter detection started with {total_segments} voiceover segments")
        passes_run = []

        # Check for LLM client
        if self.llm_client is None:
            logger.warning("No LLM client available, using fallback")
            return self._create_fallback_result(segments, overall_topic)

        # Map content type to strategy
        strategy = self._map_content_type_to_strategy(content_type)

        try:
            # Pass 1: Initial Detection
            logger.info(f"Running initial detection (strategy: {strategy})...")
            chapters = run_initial_detection(
                segments=segments,
                llm_client=self.llm_client,
                config=self.config,
                strategy=strategy,
                overall_topic=overall_topic,
            )
            passes_run.append('initial')

            if not chapters:
                logger.warning("Initial detection found no chapters, using fallback")
                return self._create_fallback_result(segments, overall_topic)

            # Detect actual content type for result
            detected_type = detect_content_type(segments, self.llm_client) if content_type == 'auto' else content_type

            # Pass 2: Boundary Refinement (if enabled)
            if self._should_run_refinement():
                logger.info("Running boundary refinement...")
                embeddings = compute_voiceover_embeddings(segments, self.config)
                chapters = run_boundary_refinement(
                    chapters=chapters,
                    segments=segments,
                    config=self.config,
                    embeddings=embeddings,
                )
                passes_run.append('refinement')

            # Pass 3: Cross-Validation (if enabled)
            if self._should_run_validation():
                logger.info("Running cross-validation...")
                chapters = run_validation(
                    chapters=chapters,
                    segments=segments,
                    llm_client=self.llm_client,
                    config=self.config,
                )
                passes_run.append('validation')

            # Pass 4: Coverage Resolution
            logger.info("Running coverage resolution...")
            chapters = run_coverage_resolution(
                chapters=chapters,
                total_segments=total_segments,
                config=self.config,
            )
            passes_run.append('coverage')

            # Resolve locations if service provided
            if location_service:
                chapters = self._resolve_chapter_locations(chapters, location_service)

            # Log detected chapters with timestamps
            for ch in chapters:
                start_time = segments[ch.start_segment_idx].get('start', ch.start_segment_idx) if ch.start_segment_idx < len(segments) else ch.start_segment_idx
                end_time = segments[ch.end_segment_idx].get('end', ch.end_segment_idx) if ch.end_segment_idx < len(segments) else ch.end_segment_idx
                logger.info(
                    f"Detected chapter: '{ch.title}' "
                    f"(segments {ch.start_segment_idx}-{ch.end_segment_idx}, "
                    f"time {start_time:.1f}s-{end_time:.1f}s, confidence: {ch.confidence:.2f})"
                )

            logger.info(f"Chapter detection complete: {len(chapters)} chapters, passes: {passes_run}")

            return DetectionResult(
                chapters=chapters,
                content_type=detected_type,
                total_segments=total_segments,
                detection_passes_run=passes_run,
                fallback_used=False,
            )

        except Exception as e:
            logger.error(f"Chapter detection failed: {e}")
            return self._create_fallback_result(segments, overall_topic, error_message=str(e))

    def _create_fallback_result(
        self,
        segments: List[Dict[str, Any]],
        overall_topic: str = None,
        error_message: str = "",
    ) -> DetectionResult:
        """Create fallback result with single chapter."""
        title = self._generate_summary_title(segments, overall_topic)

        fallback_chapter = ChapterCandidate(
            chapter_id=0,
            start_segment_idx=0,
            end_segment_idx=len(segments) - 1,
            title=title or overall_topic or 'Main Content',
            topics=[overall_topic.lower()] if overall_topic else [],
            confidence=0.3,
            detection_strategy='fallback',
            boundary_reasoning="Fallback: single chapter for entire transcript",
        )

        return DetectionResult(
            chapters=[fallback_chapter],
            content_type='general',
            total_segments=len(segments),
            detection_passes_run=['fallback'],
            fallback_used=True,
            error_message=error_message,
        )

    def _generate_summary_title(
        self,
        segments: List[Dict[str, Any]],
        overall_topic: str = None,
    ) -> str:
        """Generate a summary title for fallback chapter."""
        if overall_topic:
            return overall_topic

        if not self.llm_client:
            return "Main Content"

        # Get text sample
        text_parts = []
        total_chars = 0
        for seg in segments:
            text = seg.get('text', '')
            if total_chars + len(text) > 1500:
                break
            text_parts.append(text)
            total_chars += len(text)

        text_sample = " ".join(text_parts)

        try:
            from src.llm_client import LLMRequest, ResponseFormat

            prompt = format_summary_title_prompt(text_sample)
            request = LLMRequest(
                prompt=prompt,
                response_format=ResponseFormat.TEXT,
                max_tokens=50,
                cache_key_prefix="summary_title"
            )
            response = self.llm_client.generate(request)

            title = response.text.strip().strip('"\'')
            if title and len(title) < 100:
                return title

        except Exception as e:
            logger.debug(f"Summary title generation failed: {e}")

        return "Main Content"

    def _resolve_chapter_locations(
        self,
        chapters: List[ChapterCandidate],
        location_service: Any,
    ) -> List[ChapterCandidate]:
        """Resolve location names to GeoLocation data."""
        for chapter in chapters:
            if chapter.location_name and not chapter.location_data:
                try:
                    # Build context from visual and context keywords
                    context = " ".join(chapter.visual_keywords + chapter.context_keywords)

                    # Get co-locations from other chapters
                    co_locations = [
                        c.location_name for c in chapters
                        if c != chapter and c.location_name
                    ]

                    # Disambiguate location
                    geo_location = location_service.disambiguate(
                        chapter.location_name,
                        context=context,
                        co_locations=co_locations,
                    )

                    if geo_location:
                        chapter.location_data = geo_location.to_dict()
                        logger.info(
                            f"Resolved location: '{chapter.location_name}' -> "
                            f"{geo_location.name}, {geo_location.country_name}"
                        )

                except Exception as e:
                    logger.debug(f"Could not resolve location '{chapter.location_name}': {e}")

        return chapters

    def _should_run_refinement(self) -> bool:
        """Check if boundary refinement should run."""
        return self._get_config_value('use_boundary_refinement', True)

    def _should_run_validation(self) -> bool:
        """Check if validation pass should run."""
        return self._get_config_value('use_validation_pass', True)

    def _get_config_value(self, key: str, default: Any) -> Any:
        """Get chapter detection config value."""
        if self.config is None:
            return default

        if hasattr(self.config, 'matching'):
            matching = self.config.matching
            if hasattr(matching, 'chapter_detection'):
                ch_config = matching.chapter_detection
                if isinstance(ch_config, dict):
                    return ch_config.get(key, default)
                return getattr(ch_config, key, default)

        return default

    def _map_content_type_to_strategy(self, content_type: str) -> str:
        """Map content type to detection strategy."""
        if content_type in ('topic', 'location', 'auto'):
            return content_type

        mapping = {
            'travel': 'location',
            'educational': 'topic',
            'documentary': 'topic',
            'narrative': 'topic',
            'general': 'topic',
        }
        return mapping.get(content_type, 'topic')
