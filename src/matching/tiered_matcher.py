"""
Tiered Matcher - Two-stage matching with LLM reranking

Extracted from monolithic matching.py (Jan 2026).
Uses refactored modules to eliminate 406 lines of duplication.

Core functionality:
- Two-stage matching: embedding similarity → LLM reranking
- Smart reuse prevention and confidence adjustment
- Location-aware filtering (delegates to LocationMatcher)
- Topic-based penalty for chapter matching (uses scoring.py)
- B-roll boost and project boost (uses scoring.py)
- Alternative and secondary match generation
"""

from __future__ import annotations

import hashlib
import logging
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

# Refactored modules - REUSE instead of duplicating (406 lines saved)
from .scoring import (
    apply_topic_penalty,
    apply_broll_boost,
    apply_current_project_boost,
    calculate_adaptive_threshold,
)
from .location_matching import LocationMatcher
from .llm_providers import GeminiMatcher, ClaudeMatcher, LocalLLMMatcher

# Utils and data structures
from ..utils import (
    SRTSegment, SceneInfo, Match, AlternativeMatch, MatchResult,
    CacheManager, ReuseTracker
)
from ..topic_extraction import VideoTopics, LocationChapter
from ..location_service import GeoLocation, create_location_service
from ..keyword_extractor import find_keyword_matches
from ..face_detection import apply_face_preference
from ..logger import get_global_logger
from ..config import get_config

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)


class TieredMatcher:
    """
    Two-stage matcher with embedding search + LLM reranking.

    Uses composition with refactored modules:
    - scoring.py for confidence adjustments
    - location_matching.py for geographic filtering
    - llm_providers.py for LLM interactions
    """

    def __init__(self, config: Optional['Config'] = None, cache: Optional[CacheManager] = None, video_topics: Optional[Dict[str, VideoTopics]] = None):
        """
        Initialize TieredMatcher.

        Args:
            config: Configuration object (uses get_config() if None)
            cache: Cache manager for LLM responses
            video_topics: Dict mapping video paths to VideoTopics for chapter matching
        """
        self.config = config or get_config()
        self.cache = cache
        self.video_topics = video_topics or {}
        mc = self.config.matching

        # Matching thresholds
        self.primary_model = mc.gemini_model
        self.min_confidence = mc.min_confidence
        self.embedding_candidates = mc.embedding_candidates
        self.high_conf_threshold = mc.high_confidence_threshold
        self.low_conf_threshold = mc.low_confidence_threshold
        self.max_clip_reuse = mc.max_clip_reuse
        self.reuse_penalty = mc.reuse_penalty

        # Chapter-based matching
        self.chapter_matching_enabled = getattr(mc, 'chapter_matching_enabled', False)
        self.topic_mismatch_penalty = getattr(mc, 'topic_mismatch_penalty', 0.15)

        # Location-aware matching (compose LocationMatcher)
        self.location_matching_config = getattr(mc, 'location_matching', None)
        self.location_matching_enabled = False
        self.location_service = None
        self.location_matcher = None  # Will be LocationMatcher instance

        if self.location_matching_config:
            if isinstance(self.location_matching_config, dict):
                self.location_matching_enabled = self.location_matching_config.get('enabled', False)
            else:
                self.location_matching_enabled = getattr(self.location_matching_config, 'enabled', False)

            if self.location_matching_enabled:
                try:
                    self.location_service = create_location_service(self.config)
                    self.location_matcher = LocationMatcher(self.location_service)
                    logger.info("Location-aware matching enabled")
                except Exception as e:
                    logger.warning(f"Could not initialize location service: {e}")
                    self.location_matching_enabled = False

        # Initialize reuse tracker
        self.reuse_tracker = ReuseTracker(
            max_reuse=mc.max_clip_reuse,
            reuse_penalty=mc.reuse_penalty,
            max_source_file_reuse=getattr(mc, 'max_source_file_reuse', 0),
            source_file_penalty=getattr(mc, 'source_file_penalty', 0.05)
        )

        # Face preference (for face detection filtering)
        self.face_preference = getattr(mc, 'face_preference', 'neutral')

        # Initialize LLM providers
        self.primary_provider = None
        self.secondary_provider = None
        self.local_provider = None
        self._init_providers()

    def _init_providers(self):
        """Initialize LLM providers based on config"""
        mc = self.config.matching

        # Get model names
        gemini_model = getattr(mc, 'gemini_model', 'gemini-2.0-flash')
        anthropic_model = getattr(mc, 'anthropic_model', 'claude-3-haiku-20240307')
        ollama_model = getattr(mc, 'ollama_model', 'llama3.2')
        ollama_host = getattr(mc, 'ollama_host', 'http://localhost:11434')

        # Primary provider
        if mc.primary_provider == "gemini" and self.config.gemini_api_key:
            self.primary_provider = GeminiMatcher(self.config.gemini_api_key, gemini_model)
            logger.info(f"Primary LLM: Gemini ({gemini_model})")
        elif mc.primary_provider == "anthropic" and self.config.anthropic_api_key:
            self.primary_provider = ClaudeMatcher(self.config.anthropic_api_key, anthropic_model)
            logger.info(f"Primary LLM: Claude ({anthropic_model})")
        elif self.config.gemini_api_key:
            self.primary_provider = GeminiMatcher(self.config.gemini_api_key, gemini_model)
            logger.info(f"Primary LLM: Gemini ({gemini_model}) (auto)")
        elif self.config.anthropic_api_key:
            self.primary_provider = ClaudeMatcher(self.config.anthropic_api_key, anthropic_model)
            logger.info(f"Primary LLM: Claude ({anthropic_model}) (auto)")

        # Secondary provider
        if mc.secondary_provider == "anthropic" and self.config.anthropic_api_key:
            self.secondary_provider = ClaudeMatcher(self.config.anthropic_api_key, anthropic_model)
            logger.info(f"Secondary LLM: Claude ({anthropic_model})")
        elif mc.secondary_provider == "gemini" and self.config.gemini_api_key:
            self.secondary_provider = GeminiMatcher(self.config.gemini_api_key, gemini_model)
            logger.info(f"Secondary LLM: Gemini ({gemini_model})")

        # Local provider
        if mc.use_local_for_review:
            try:
                self.local_provider = LocalLLMMatcher(ollama_model, ollama_host)
                import requests
                requests.get(f"{ollama_host}/api/tags", timeout=2)
                logger.info(f"Local LLM: Ollama ({ollama_model})")
            except:
                logger.info("Local LLM: Not available (Ollama not running)")
                self.local_provider = None

    # Location methods - delegate to LocationMatcher
    def set_location_chapters(self, location_chapters: List[LocationChapter]):
        """Set location chapters for location-aware matching"""
        if self.location_matcher:
            self.location_matcher.set_location_chapters(location_chapters)
            logger.info(f"Set {len(location_chapters)} location chapters")

    def set_video_locations(self, video_locations: Dict[str, GeoLocation]):
        """Set video location data for location-aware matching"""
        if self.location_matcher:
            self.location_matcher.set_video_locations(video_locations)
            logger.info(f"Set locations for {len(video_locations)} videos")

    # Utility methods
    def _calculate_confidence_variance(self, candidates: List[Tuple[SRTSegment, float]], top_n: int = 5) -> float:
        """
        Calculate standard deviation of top-N candidate similarity scores.

        High variance (> 0.1) indicates uncertain match - multiple candidates have
        similar scores, making the selection less definitive.
        Low variance indicates clear winner with others scoring much lower.

        Args:
            candidates: List of (video_segment, similarity) tuples
            top_n: Number of top candidates to consider (default: 5)

        Returns:
            Standard deviation of top-N similarity scores (0.0 if < 2 candidates)
        """
        if len(candidates) < 2:
            return 0.0

        # Get top-N similarity scores
        top_scores = [sim for _, sim in candidates[:top_n]]

        if len(top_scores) < 2:
            return 0.0

        try:
            return statistics.stdev(top_scores)
        except statistics.StatisticsError:
            return 0.0

    def _extract_matched_keywords(
        self,
        vo_segment: SRTSegment,
        video_segment: SRTSegment
    ) -> List[str]:
        """
        Extract common keywords between voiceover and video transcript.

        Finds keywords that appear in both the voiceover segment and the selected
        video's transcript/keywords. Returns unique matched keywords sorted by
        frequency of occurrence.

        Args:
            vo_segment: Voiceover segment with text and optional keywords
            video_segment: Selected video segment with text and optional keywords

        Returns:
            List of matched keywords (lowercase, deduplicated)
        """
        matched = set()

        # Get voiceover keywords from both keywords list and text
        vo_keywords = set()
        if hasattr(vo_segment, 'keywords') and vo_segment.keywords:
            vo_keywords.update(kw.lower().strip() for kw in vo_segment.keywords if kw)

        # Extract significant words from voiceover text (>= 4 chars, not common words)
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                        'can', 'her', 'was', 'one', 'our', 'out', 'has', 'have',
                        'been', 'from', 'this', 'that', 'with', 'they', 'what',
                        'will', 'there', 'their', 'about', 'would', 'which', 'into'}
        if vo_segment.text:
            words = vo_segment.text.lower().split()
            vo_keywords.update(
                w.strip('.,!?:;"\'()[]{}') for w in words
                if len(w) >= 4 and w.lower() not in common_words
            )

        # Get video keywords from both keywords list and text
        video_keywords = set()
        if hasattr(video_segment, 'keywords') and video_segment.keywords:
            video_keywords.update(kw.lower().strip() for kw in video_segment.keywords if kw)

        # Extract significant words from video text
        if video_segment.text:
            words = video_segment.text.lower().split()
            video_keywords.update(
                w.strip('.,!?:;"\'()[]{}') for w in words
                if len(w) >= 4 and w.lower() not in common_words
            )

        # Find intersection
        matched = vo_keywords & video_keywords

        # Filter out very short keywords and return sorted list
        result = sorted([kw for kw in matched if len(kw) >= 3])

        return result

    def _should_skip_llm(self, similarity: float) -> bool:
        """Skip LLM if embedding similarity is high enough"""
        return similarity >= self.config.matching.high_confidence_threshold

    def _get_cache_key(self, vo_text: str, candidates: List[Tuple[SRTSegment, float]]) -> str:
        """Generate cache key for LLM response"""
        content = vo_text + "|" + "|".join(c[0].text for c in candidates[:5])
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _get_cached_response(self, cache_key: str) -> Optional[Tuple[int, float, str]]:
        """Get cached LLM response"""
        if not self.config.matching.cache_llm_responses or not self.cache:
            return None

        cached = self.cache.get_llm_response(cache_key)
        if cached:
            return (cached['selected'], cached['confidence'], cached['reasoning'])
        return None

    def _cache_response(self, cache_key: str, selected: int, confidence: float, reasoning: str):
        """Cache LLM response"""
        if self.config.matching.cache_llm_responses and self.cache:
            self.cache.save_llm_response(cache_key, {
                'selected': selected,
                'confidence': confidence,
                'reasoning': reasoning
            })

    def _get_scene_for_segment(
        self,
        segment: SRTSegment,
        scenes: Optional[Dict[str, List[SceneInfo]]]
    ) -> Optional[SceneInfo]:
        """Find the scene containing this segment"""
        if not scenes:
            return None

        video_scenes = scenes.get(segment.source_file, [])

        for scene in video_scenes:
            if scene.start_time <= segment.start_time < scene.end_time:
                return scene

        return None

    def _build_context(
        self,
        context_before: Optional[List[SRTSegment]],
        context_after: Optional[List[SRTSegment]]
    ) -> Optional[str]:
        """Build context string from surrounding segments"""
        if not context_before and not context_after:
            return None

        parts = []

        if context_before:
            before_text = " | ".join(s.text[:50] for s in context_before[-2:])
            parts.append(f"Before: {before_text}")

        if context_after:
            after_text = " | ".join(s.text[:50] for s in context_after[:2])
            parts.append(f"After: {after_text}")

        return " || ".join(parts)

    def match_segment(
        self,
        vo_segment: SRTSegment,
        candidates: List[Tuple[SRTSegment, float]],
        scenes: Optional[Dict[str, List[SceneInfo]]] = None,
        context_before: Optional[List[SRTSegment]] = None,
        context_after: Optional[List[SRTSegment]] = None,
        segment_idx: int = 0
    ) -> MatchResult:
        """
        Match a single voiceover segment.
        Returns MatchResult with primary match and alternatives.

        Args:
            vo_segment: Voiceover segment to match
            candidates: List of (video_segment, similarity) tuples
            scenes: Optional scene info for videos
            context_before: Previous voiceover segments for context
            context_after: Following voiceover segments for context
            segment_idx: Index of voiceover segment (for location chapter lookup)
        """
        mc = self.config.matching

        logger.info(f"  match_segment: entering for '{vo_segment.text[:30]}...'")

        # Guard: return gap if no candidates
        if not candidates:
            logger.warning(f"  match_segment: no candidates for '{vo_segment.text[:30]}...'")
            gap_match = Match(
                voiceover_segment=vo_segment,
                video_segment=vo_segment,
                video_scene=None,
                confidence=0.0,
                reasoning="No video candidates available"
            )
            return MatchResult(primary_match=gap_match, has_gap=True, gap_reason="No candidates")

        # Apply face preference if set
        if self.face_preference != 'neutral':
            current_project_candidates = []
            global_cache_candidates = []

            for seg, sim in candidates:
                source = getattr(seg, 'source', None)
                if source == 'global_cache':
                    global_cache_candidates.append((seg, sim))
                else:
                    current_project_candidates.append((seg, sim))

            # Apply face detection to current project videos
            if current_project_candidates:
                logger.info(f"  match_segment: face preference '{self.face_preference}' - {len(current_project_candidates)} project, {len(global_cache_candidates)} cached")
                cache_dir = self.cache.cache_dir if hasattr(self.cache, 'cache_dir') else None
                current_project_candidates = apply_face_preference(current_project_candidates, self.face_preference, cache_dir)

            # Apply cached face scores to global cache candidates
            if global_cache_candidates:
                adjusted_cache = []
                for seg, score in global_cache_candidates:
                    cached_face_score = getattr(seg, 'face_score', None)
                    if cached_face_score is not None:
                        if self.face_preference == "more":
                            boost = cached_face_score * 0.3
                            adjusted_cache.append((seg, min(1.0, score + boost)))
                        elif self.face_preference == "none":
                            boost = (1.0 - cached_face_score) * 0.3
                            adjusted_cache.append((seg, min(1.0, score + boost)))
                        else:
                            adjusted_cache.append((seg, score))
                    else:
                        adjusted_cache.append((seg, score))
                global_cache_candidates = adjusted_cache

            candidates = current_project_candidates + global_cache_candidates
            candidates.sort(key=lambda x: x[1], reverse=True)

        # Apply location-based filtering (delegate to LocationMatcher)
        location_filter_applied = False
        location_reason = ""
        if self.location_matcher:
            candidates, location_filter_applied, location_reason = self.location_matcher.apply_location_filter(
                vo_segment, candidates, segment_idx,
                self.location_matching_enabled, self.location_matching_config
            )

        # Apply smart reuse filtering
        valid_candidates = []
        for seg, sim in candidates:
            if self.reuse_tracker.can_use(seg):
                adjusted_sim = self.reuse_tracker.adjust_confidence(seg, sim)
                valid_candidates.append((seg, adjusted_sim))

        if not valid_candidates:
            valid_candidates = [(seg, sim * 0.5) for seg, sim in candidates[:5]]

        if not valid_candidates:
            logger.warning(f"  match_segment: no valid candidates after filtering")
            gap_match = Match(
                voiceover_segment=vo_segment,
                video_segment=vo_segment,
                video_scene=None,
                confidence=0.0,
                reasoning="No valid candidates after filtering"
            )
            return MatchResult(primary_match=gap_match, has_gap=True, gap_reason="All candidates filtered")

        # Check for high-confidence embedding match
        top_similarity = valid_candidates[0][1] if valid_candidates else 0

        # Calculate adaptive threshold if enabled
        adaptive_threshold_enabled = getattr(mc, 'adaptive_threshold_enabled', True)
        if adaptive_threshold_enabled:
            skip_threshold, threshold_reason = calculate_adaptive_threshold(
                base_threshold=mc.skip_llm_threshold,
                voiceover_text=vo_segment.text,
                candidates=valid_candidates,
                config=self.config
            )
            logger.info(f"  match_segment: top_sim={top_similarity:.3f}, adaptive_threshold={skip_threshold:.3f} ({threshold_reason})")
        else:
            skip_threshold = mc.skip_llm_threshold
            logger.info(f"  match_segment: top_sim={top_similarity:.3f}, skip_threshold={skip_threshold}")

        if top_similarity >= skip_threshold:
            best_seg = valid_candidates[0][0]
            self.reuse_tracker.record_usage(best_seg)

            scene = self._get_scene_for_segment(best_seg, scenes)

            # Apply scoring adjustments using refactored functions
            adjusted_confidence, topic_penalty_reason = apply_topic_penalty(
                top_similarity, vo_segment, best_seg,
                video_topics=self.video_topics,
                chapter_matching_enabled=self.chapter_matching_enabled,
                topic_mismatch_penalty=self.topic_mismatch_penalty
            )

            adjusted_confidence, broll_reason = apply_broll_boost(
                adjusted_confidence, best_seg, self.config
            )

            adjusted_confidence, project_reason = apply_current_project_boost(
                adjusted_confidence, best_seg, self.config
            )

            reasoning = f"High embedding similarity ({top_similarity:.2f})"
            if topic_penalty_reason:
                reasoning += f" [{topic_penalty_reason}]"
            if broll_reason:
                reasoning += f" [{broll_reason}]"
            if project_reason:
                reasoning += f" [{project_reason}]"

            match = Match(
                voiceover_segment=vo_segment,
                video_segment=best_seg,
                video_scene=scene,
                confidence=adjusted_confidence,
                reasoning=reasoning,
                embedding_similarity=top_similarity,
                clip_reuse_count=self.reuse_tracker.get_usage_count(best_seg)
            )

            alternatives = self._get_alternatives(valid_candidates[1:4], scenes, best_seg)

            used_video_files = {best_seg.source_file}
            alt_segments = []
            for alt in alternatives:
                used_video_files.add(alt.video_segment.source_file)
                alt_segments.append(alt.video_segment)

            secondary_matches = self._get_secondary_matches(
                valid_candidates, scenes, used_video_files,
                primary_segment=best_seg, alt_segments=alt_segments
            )

            # Calculate confidence variance for top candidates
            confidence_variance = self._calculate_confidence_variance(valid_candidates)

            # Extract matched keywords between voiceover and selected video
            matched_keywords = self._extract_matched_keywords(vo_segment, best_seg)

            return MatchResult(
                primary_match=match,
                alternatives=alternatives,
                secondary_matches=secondary_matches,
                confidence_variance=confidence_variance,
                matched_keywords=matched_keywords
            )

        # Check cache
        cache_key = self._get_cache_key(vo_segment.text, valid_candidates)
        cached = self._get_cached_response(cache_key)

        if cached:
            selected_idx, confidence, reasoning = cached
            selected_idx = min(selected_idx, len(valid_candidates) - 1)
            cached_seg = valid_candidates[selected_idx][0]

            if self.reuse_tracker.can_use(cached_seg):
                self.reuse_tracker.record_usage(cached_seg)
                scene = self._get_scene_for_segment(cached_seg, scenes)

                # Apply scoring adjustments
                adjusted_confidence, topic_penalty_reason = apply_topic_penalty(
                    confidence, vo_segment, cached_seg,
                    video_topics=self.video_topics,
                    chapter_matching_enabled=self.chapter_matching_enabled,
                    topic_mismatch_penalty=self.topic_mismatch_penalty
                )

                adjusted_confidence, broll_reason = apply_broll_boost(
                    adjusted_confidence, cached_seg, self.config
                )

                adjusted_confidence, project_reason = apply_current_project_boost(
                    adjusted_confidence, cached_seg, self.config
                )

                final_reasoning = f"(cached) {reasoning}"
                if topic_penalty_reason:
                    final_reasoning += f" [{topic_penalty_reason}]"
                if broll_reason:
                    final_reasoning += f" [{broll_reason}]"
                if project_reason:
                    final_reasoning += f" [{project_reason}]"

                match = Match(
                    voiceover_segment=vo_segment,
                    video_segment=cached_seg,
                    video_scene=scene,
                    confidence=adjusted_confidence,
                    reasoning=final_reasoning,
                    embedding_similarity=valid_candidates[selected_idx][1],
                    clip_reuse_count=self.reuse_tracker.get_usage_count(cached_seg)
                )

                alternatives = self._get_alternatives(
                    [c for i, c in enumerate(valid_candidates[:4]) if i != selected_idx],
                    scenes, cached_seg
                )

                used_video_files = {cached_seg.source_file}
                alt_segments = []
                for alt in alternatives:
                    used_video_files.add(alt.video_segment.source_file)
                    alt_segments.append(alt.video_segment)

                secondary_matches = self._get_secondary_matches(
                    valid_candidates, scenes, used_video_files,
                    primary_segment=cached_seg, alt_segments=alt_segments
                )

                # Calculate confidence variance for top candidates
                confidence_variance = self._calculate_confidence_variance(valid_candidates)

                # Extract matched keywords between voiceover and selected video
                matched_keywords = self._extract_matched_keywords(vo_segment, cached_seg)

                return MatchResult(
                    primary_match=match,
                    alternatives=alternatives,
                    secondary_matches=secondary_matches,
                    confidence_variance=confidence_variance,
                    matched_keywords=matched_keywords
                )

        # Build context and use LLM
        context = self._build_context(context_before, context_after)
        negative_rules = self.config.negative_matching.rules if self.config.negative_matching.enabled else None

        provider = self.primary_provider

        if provider:
            logger.info(f"  match_segment: calling {type(provider).__name__}.match_batch()...")
            try:
                results = provider.match_batch(
                    [(vo_segment.text, valid_candidates[:5])],
                    context=context,
                    negative_rules=negative_rules
                )
                logger.info(f"  match_segment: LLM returned results")
                selected_idx, confidence, reasoning = results[0]

                # Check if ambiguous - use secondary provider
                if confidence < mc.ambiguous_threshold and self.secondary_provider:
                    logger.debug(f"Ambiguous match ({confidence:.2f}), using secondary LLM")
                    secondary_results = self.secondary_provider.match_batch(
                        [(vo_segment.text, valid_candidates[:5])],
                        context=context,
                        negative_rules=negative_rules
                    )
                    sec_idx, sec_conf, sec_reason = secondary_results[0]

                    if sec_conf > confidence:
                        selected_idx, confidence, reasoning = sec_idx, sec_conf, f"(secondary) {sec_reason}"

                self._cache_response(cache_key, selected_idx, confidence, reasoning)

            except Exception as e:
                logger.warning(f"LLM matching failed: {e}")
                selected_idx = 0
                embedding_sim = valid_candidates[0][1]
                confidence = 0.60
                reasoning = f"LLM fallback (emb_sim={embedding_sim:.2f})"
        else:
            selected_idx = 0
            embedding_sim = valid_candidates[0][1]
            confidence = 0.60
            reasoning = f"Embedding similarity only (sim={embedding_sim:.2f})"

        # Build result
        selected_idx = min(selected_idx, len(valid_candidates) - 1)
        best_seg = valid_candidates[selected_idx][0]
        self.reuse_tracker.record_usage(best_seg)

        scene = self._get_scene_for_segment(best_seg, scenes)

        # Check for keyword/visual matches
        vo_keywords = getattr(vo_segment, 'keywords', []) or []
        seg_keywords = getattr(best_seg, 'keywords', []) or []
        keyword_boost, is_kw_match, is_vis_match = find_keyword_matches(
            vo_keywords,
            seg_keywords,
            scene.visual_keywords if scene else None
        )

        # Apply scoring adjustments
        base_confidence = min(1.0, confidence + keyword_boost)
        adjusted_confidence, topic_penalty_reason = apply_topic_penalty(
            base_confidence, vo_segment, best_seg,
            video_topics=self.video_topics,
            chapter_matching_enabled=self.chapter_matching_enabled,
            topic_mismatch_penalty=self.topic_mismatch_penalty
        )

        adjusted_confidence, broll_reason = apply_broll_boost(
            adjusted_confidence, best_seg, self.config
        )

        adjusted_confidence, project_reason = apply_current_project_boost(
            adjusted_confidence, best_seg, self.config
        )

        final_reasoning = reasoning
        if topic_penalty_reason:
            final_reasoning += f" [{topic_penalty_reason}]"
        if broll_reason:
            final_reasoning += f" [{broll_reason}]"
        if project_reason:
            final_reasoning += f" [{project_reason}]"

        match = Match(
            voiceover_segment=vo_segment,
            video_segment=best_seg,
            video_scene=scene,
            confidence=adjusted_confidence,
            reasoning=final_reasoning,
            is_keyword_match=is_kw_match,
            is_visual_match=is_vis_match,
            embedding_similarity=valid_candidates[selected_idx][1],
            clip_reuse_count=self.reuse_tracker.get_usage_count(best_seg)
        )

        # Log match decision
        run_logger = get_global_logger()
        if run_logger:
            run_logger.log_match_decision(
                segment_index=segment_idx,
                voiceover_text=vo_segment.text,
                selected_clip=best_seg.source_file,
                confidence=adjusted_confidence,
                reasoning=final_reasoning,
                embedding_similarity=valid_candidates[selected_idx][1],
                duration_penalty=0.0,
                keyword_boost=keyword_boost,
                entity_boost=0.0,
                is_keyword_match=is_kw_match,
                is_visual_match=is_vis_match,
                alternatives_considered=len(candidates),
                llm_reranked=True if self.primary_provider else False,
                clip_reuse_count=self.reuse_tracker.get_usage_count(best_seg)
            )

        # Get alternatives and secondary matches
        alternatives = self._get_alternatives(
            [c for i, c in enumerate(valid_candidates[:4]) if i != selected_idx],
            scenes, best_seg
        )

        used_video_files = {best_seg.source_file}
        alt_segments = []
        for alt in alternatives:
            used_video_files.add(alt.video_segment.source_file)
            alt_segments.append(alt.video_segment)

        secondary_matches = self._get_secondary_matches(
            valid_candidates, scenes, used_video_files,
            primary_segment=best_seg, alt_segments=alt_segments
        )

        # Check for gap
        has_gap = confidence < self.config.matching.confidence_threshold
        gap_reason = f"Low confidence ({confidence:.2f})" if has_gap else ""

        logger.debug(f"Match decision for segment: '{vo_segment.text[:50]}...'")
        logger.debug(f"  Video: {Path(best_seg.source_file).name} @ {best_seg.start_time:.1f}s")
        logger.debug(f"  Confidence: {confidence:.2f}")
        logger.debug(f"  Reason: {reasoning[:100]}...")
        logger.debug(f"  Alternatives: {len(alternatives)}, Has gap: {has_gap}")

        # Calculate confidence variance for top candidates
        confidence_variance = self._calculate_confidence_variance(valid_candidates)

        # Extract matched keywords between voiceover and selected video
        matched_keywords = self._extract_matched_keywords(vo_segment, best_seg)

        return MatchResult(
            primary_match=match,
            alternatives=alternatives,
            secondary_matches=secondary_matches,
            has_gap=has_gap,
            gap_reason=gap_reason,
            confidence_variance=confidence_variance,
            matched_keywords=matched_keywords
        )

    def _get_alternatives(
        self,
        candidates: List[Tuple[SRTSegment, float]],
        scenes: Optional[Dict[str, List[SceneInfo]]],
        primary_match: Optional[SRTSegment] = None
    ) -> List[AlternativeMatch]:
        """Get alternative matches, preferring different sources from primary"""
        alternatives = []
        used_sources = set()

        if primary_match and primary_match.source_file:
            used_sources.add(primary_match.source_file)

        # First pass: prefer different sources
        for seg, sim in candidates:
            if len(alternatives) >= self.config.output.num_alternatives:
                break

            if seg.source_file not in used_sources:
                scene = self._get_scene_for_segment(seg, scenes)
                alternatives.append(AlternativeMatch(
                    video_segment=seg,
                    video_scene=scene,
                    confidence=sim,
                    reasoning=f"Alternative (different source: {Path(seg.source_file).stem})"
                ))
                used_sources.add(seg.source_file)

        # Second pass: fill remaining slots
        if len(alternatives) < self.config.output.num_alternatives:
            for seg, sim in candidates:
                if len(alternatives) >= self.config.output.num_alternatives:
                    break

                if any(alt.video_segment.source_file == seg.source_file and
                       alt.video_segment.start_time == seg.start_time for alt in alternatives):
                    continue

                scene = self._get_scene_for_segment(seg, scenes)
                alternatives.append(AlternativeMatch(
                    video_segment=seg,
                    video_scene=scene,
                    confidence=sim * 0.9,
                    reasoning="Alternative (fallback)"
                ))

        return alternatives

    def _get_secondary_matches(
        self,
        candidates: List[Tuple[SRTSegment, float]],
        scenes: Optional[Dict[str, List[SceneInfo]]],
        excluded_video_files: set,
        primary_segment: Optional[SRTSegment] = None,
        alt_segments: Optional[List[SRTSegment]] = None
    ) -> List[AlternativeMatch]:
        """
        Get secondary matches for V4-V6.

        Three-pass approach:
        1. Different video files from V1-V3, different from each other
        2. Different video files from V1-V3, allow same source within V4-V6
        3. Allow same video file as V1-V3 but different segment
        """
        secondary = []
        used_sources = set()
        num_secondary = 3

        # Collect exact segments used by V1-V3
        used_segments = set()
        if primary_segment:
            used_segments.add((primary_segment.source_file, primary_segment.start_time))
        if alt_segments:
            for seg in alt_segments:
                if seg:
                    used_segments.add((seg.source_file, seg.start_time))

        # First pass: Different video files, different from each other
        for seg, sim in candidates:
            if len(secondary) >= num_secondary:
                break

            if seg.source_file in excluded_video_files:
                continue

            if seg.source_file in used_sources:
                continue

            scene = self._get_scene_for_segment(seg, scenes)
            position = len(secondary)
            label = "Secondary Primary" if position == 0 else f"Secondary Alt {position}"

            secondary.append(AlternativeMatch(
                video_segment=seg,
                video_scene=scene,
                confidence=sim,
                reasoning=f"{label} (source: {Path(seg.source_file).stem})"
            ))
            used_sources.add(seg.source_file)

        # Second pass: Different video files, allow same source within V4-V6
        if len(secondary) < num_secondary:
            for seg, sim in candidates:
                if len(secondary) >= num_secondary:
                    break

                if seg.source_file in excluded_video_files:
                    continue

                if any(s.video_segment.source_file == seg.source_file and
                       s.video_segment.start_time == seg.start_time for s in secondary):
                    continue

                scene = self._get_scene_for_segment(seg, scenes)
                position = len(secondary)
                label = "Secondary Primary" if position == 0 else f"Secondary Alt {position}"

                secondary.append(AlternativeMatch(
                    video_segment=seg,
                    video_scene=scene,
                    confidence=sim * 0.95,
                    reasoning=f"{label} (same source ok)"
                ))

        # Third pass: Allow same video file but different segment
        if len(secondary) < num_secondary:
            for seg, sim in candidates:
                if len(secondary) >= num_secondary:
                    break

                seg_key = (seg.source_file, seg.start_time)
                if seg_key in used_segments:
                    continue

                if any(s.video_segment.source_file == seg.source_file and
                       s.video_segment.start_time == seg.start_time for s in secondary):
                    continue

                scene = self._get_scene_for_segment(seg, scenes)
                position = len(secondary)
                label = "Secondary Primary" if position == 0 else f"Secondary Alt {position}"

                secondary.append(AlternativeMatch(
                    video_segment=seg,
                    video_scene=scene,
                    confidence=sim * 0.85,
                    reasoning=f"{label} (fallback - different segment)"
                ))

        return secondary

    def review_with_local_llm(self, matches: List[MatchResult]) -> List[MatchResult]:
        """Use local LLM to review and potentially adjust low-confidence matches"""
        if not self.local_provider:
            return matches

        low_confidence = [
            (i, m) for i, m in enumerate(matches)
            if m.primary_match.confidence < self.config.matching.ambiguous_threshold
        ]

        if not low_confidence:
            return matches

        logger.info(f"Reviewing {len(low_confidence)} low-confidence matches with local LLM...")

        for idx, match_result in low_confidence:
            primary = match_result.primary_match

            candidates = [
                (primary.video_segment, primary.confidence),
                *[(alt.video_segment, alt.confidence) for alt in match_result.alternatives]
            ]

            try:
                results = self.local_provider.match_batch(
                    [(primary.voiceover_segment.text, candidates)]
                )
                new_idx, new_conf, new_reason = results[0]

                if new_conf > primary.confidence:
                    logger.debug(f"Local LLM improved match: {primary.confidence:.2f} -> {new_conf:.2f}")

                    if new_idx == 0:
                        primary.confidence = new_conf
                        primary.reasoning = f"(local refined) {new_reason}"
                    else:
                        new_seg = candidates[new_idx][0]
                        match_result.primary_match = Match(
                            voiceover_segment=primary.voiceover_segment,
                            video_segment=new_seg,
                            video_scene=match_result.alternatives[new_idx-1].video_scene if new_idx <= len(match_result.alternatives) else None,
                            confidence=new_conf,
                            reasoning=f"(local selected) {new_reason}",
                            embedding_similarity=candidates[new_idx][1]
                        )
            except Exception as e:
                logger.debug(f"Local LLM review failed: {e}")

        return matches


__all__ = ['TieredMatcher']
